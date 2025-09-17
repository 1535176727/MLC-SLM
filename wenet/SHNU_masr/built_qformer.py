import torch
import torch.nn as nn
from wenet.SHNU_masr.Qformer import BertConfig, BertLMHeadModel
def build_audio_qformer(num_query_token: int, vision_width: int, num_hidden_layers: int = 2):

    encoder_config = BertConfig()
    encoder_config.num_hidden_layers = num_hidden_layers
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = 1

    Qformer = BertLMHeadModel(config=encoder_config)
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    
    return Qformer, query_tokens

import torch.nn.functional as F

def windows_level_qformer(
    encoder_out: torch.Tensor,
    attention_mask: torch.Tensor,
    qformer: nn.Module,
    query_tokens: nn.Parameter,
    final_projector: nn.Module,
) -> torch.Tensor:

    batch, _, dim = encoder_out.shape
    kernel = (1, 17)
    stride = (1, 17)
    audio_embeds_new = F.unfold(encoder_out.transpose(1, 2).unsqueeze(2),
                                kernel_size=kernel, stride=stride)
    audio_embeds_new = audio_embeds_new.view(batch, dim, kernel[1], -1)
    audio_embeds_new = torch.permute(audio_embeds_new, [0, 3, 2, 1])
    windowed_audio_embeds = audio_embeds_new.reshape(-1, kernel[1], dim)

    mask_4d = attention_mask.float().unsqueeze(1).unsqueeze(2)
    unfolded_mask = F.unfold(mask_4d, kernel_size=kernel, stride=stride)
    num_windows = windowed_audio_embeds.shape[0] // batch
    windowed_attention_mask = unfolded_mask.view(batch, num_windows, kernel[1]).reshape(-1, kernel[1])
    

    num_query_tokens = query_tokens.shape[1]
    expanded_query_tokens = query_tokens.expand(windowed_audio_embeds.shape[0], -1, -1)
    
    extended_attention_mask = windowed_attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    qformer_output = qformer(
        query_embeds=expanded_query_tokens,
        encoder_hidden_states=windowed_audio_embeds.to(torch.float32),
        encoder_attention_mask=extended_attention_mask.to(torch.float32)
    )

    qformer_output_reshaped = qformer_output.last_hidden_state.view(
        batch, num_windows, num_query_tokens, -1
    )
    final_qformer_output = qformer_output_reshaped.flatten(start_dim=1, end_dim=2)

    speech_embeds = final_projector(final_qformer_output)

    return speech_embeds