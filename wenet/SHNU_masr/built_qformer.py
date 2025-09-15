import torch
import torch.nn as nn
from wenet.SHNU_masr.Qformer import BertConfig, BertLMHeadModel
def build_audio_qformer(num_query_token: int, vision_width: int, num_hidden_layers: int = 2):
    """
    构建Q-Former模型和可学习的query tokens。
    
    Args:
        num_query_token (int): 查询向量的数量，例如 32。
        vision_width (int): 输入的音频特征维度。
        num_hidden_layers (int): Q-Former的层数。

    Returns:
        Tuple[BertLMHeadModel, nn.Parameter]: Q-Former模型和query_tokens。
    """
    # Q-Former的配置
    encoder_config = BertConfig()
    encoder_config.num_hidden_layers = num_hidden_layers
    encoder_config.encoder_width = vision_width  # 输入特征的维度
    encoder_config.add_cross_attention = True    # 开启交叉注意力
    encoder_config.cross_attention_freq = 1      # 每层都有交叉注意力
    
    # 初始化Q-Former模型
    Qformer = BertLMHeadModel(config=encoder_config)
    
    # 初始化可学习的query tokens
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    
    return Qformer, query_tokens
import torch
import torch.nn as nn
import torch.nn.functional as F

def windows_level_qformer(
    encoder_out: torch.Tensor,
    attention_mask: torch.Tensor,
    qformer: nn.Module,
    query_tokens: nn.Parameter,
    final_projector: nn.Module,
) -> torch.Tensor:
    """
    对编码器输出执行窗口切分，并通过Q-Former进行处理和投影的独立函数。

    Args:
        encoder_out (torch.Tensor): 融合后的编码器输出，形状为 (B, T, D_in)。
        attention_mask (torch.Tensor): 对应的注意力mask，形状为 (B, T)。
        qformer (nn.Module): Q-Former模型实例。
        query_tokens (nn.Parameter): 可学习的查询向量，形状为 (1, num_tokens, D_qformer)。
        final_projector (nn.Module): 最后的线性投影层。

    Returns:
        torch.Tensor: 处理后准备送入LLM的语音嵌入，形状为 (B, T_out, D_llm)。
    """
    # 1. 窗口化音频特征
    batch, _, dim = encoder_out.shape
    kernel = (1, 17)
    stride = (1, 17)
    audio_embeds_new = F.unfold(encoder_out.transpose(1, 2).unsqueeze(2),
                                kernel_size=kernel, stride=stride)
    audio_embeds_new = audio_embeds_new.view(batch, dim, kernel[1], -1)
    audio_embeds_new = torch.permute(audio_embeds_new, [0, 3, 2, 1])
    windowed_audio_embeds = audio_embeds_new.reshape(-1, kernel[1], dim)

    # 2. 窗口化对应的 Mask
    mask_4d = attention_mask.float().unsqueeze(1).unsqueeze(2)
    unfolded_mask = F.unfold(mask_4d, kernel_size=kernel, stride=stride)
    num_windows = windowed_audio_embeds.shape[0] // batch
    windowed_attention_mask = unfolded_mask.view(batch, num_windows, kernel[1]).reshape(-1, kernel[1])
    
    # 3. 准备 Q-Former 输入并运行
    #    a. 从传入的 query_tokens 获取 num_query_tokens
    num_query_tokens = query_tokens.shape[1]
    #    b. 扩展 query_tokens
    expanded_query_tokens = query_tokens.expand(windowed_audio_embeds.shape[0], -1, -1)
    
    #    c. 准备 mask
    extended_attention_mask = windowed_attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    #    d. 运行 Q-Former
    qformer_output = qformer(
        query_embeds=expanded_query_tokens,
        encoder_hidden_states=windowed_audio_embeds.to(torch.float32),
        encoder_attention_mask=extended_attention_mask.to(torch.float32)
    )

    # 4. 重塑 Q-Former 输出
    qformer_output_reshaped = qformer_output.last_hidden_state.view(
        batch, num_windows, num_query_tokens, -1
    )
    final_qformer_output = qformer_output_reshaped.flatten(start_dim=1, end_dim=2)

    # 5. 最终投影
    speech_embeds = final_projector(final_qformer_output)

    return speech_embeds