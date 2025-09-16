# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2025 ASLP@NPU for MLC-SLM Baseline. (authors: Bingshen Mu)
#               2025 Shanghai Normal University (authors: Yuxiang Mei)  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import Dict, List, Optional, Tuple,Union
import torch
import torch.utils.checkpoint as checkpoint
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.search import DecodeResult
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, Qwen2AudioForConditionalGeneration
from wenet.transformer.subsampling import WhisperProjector
from transformers import AutoProcessor
from wenet.SHNU_masr.Qformer import BertConfig, BertLMHeadModel
from wenet.SHNU_masr.base_model import BaseModel
import torch.nn.functional as F
from wenet.SHNU_masr.cross_attention import CrossAttentionFusion,BiCrossAttentionFusion,GatedCrossAttentionFusion,GatedBiCrossAttentionFusion,ConcatCrossAttentionFusion,ConcatBiCrossAttentionFusion
class SHNU_masr(torch.nn.Module):
    """Encoder + Projector + Qwen LLM Decoder"""
#8393M
    def __init__(
        self,
        whisper_encoder: BaseEncoder,
        whisper_processor: AutoProcessor,
        hubert_encoder: BaseEncoder,
        hubert_processor: torch.nn.Module,
        qwen: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        use_qwen_lora: bool,
    ):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.whisper_processor = whisper_processor
        self.hubert_encoder = hubert_encoder
        self.hubert_processor = hubert_processor
        self.use_qwen_lora = use_qwen_lora
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.cross_type = "whisper_mhubert"  # or "bicross_attn"
        self.projector_type = "qformer"  # or "conv"
        if self.cross_type == "cross_attn":
            self.cross_attn = CrossAttentionFusion(query_dim=1280, kv_dim=768, num_heads=8)
            self.encoder_projector = WhisperProjector(downsample_rate=4, idim=1280, odim=3584)
        elif self.cross_type == "bicross_attn":
            self.bicross_attn = BiCrossAttentionFusion(whisper_dim=1280, mhubert_dim=768, num_heads=8)
            self.encoder_projector = WhisperProjector(downsample_rate=4, idim=2048, odim=3584)
        elif self.cross_type == "gated_cross_attn":
            self.cross_attn = GatedCrossAttentionFusion(query_dim=1280, kv_dim=768, num_heads=8)
            self.encoder_projector = WhisperProjector(downsample_rate=4, idim=1280, odim=3584)
        elif self.cross_type == "gated_bicross_attn":
            self.bicross_attn = GatedBiCrossAttentionFusion(whisper_dim=1280, mhubert_dim=768, num_heads=8)
            self.encoder_projector = WhisperProjector(downsample_rate=4, idim=2048, odim=3584)
        elif self.cross_type == "concat_bicross_attn":
            self.bicross_attn = ConcatBiCrossAttentionFusion(whisper_dim=1280, mhubert_dim=768, num_heads=8)
            self.encoder_projector = WhisperProjector(downsample_rate=4, idim=4096, odim=3584)
        elif self.cross_type == "concat_cross_attn":
            self.cross_attn = ConcatCrossAttentionFusion(whisper_dim=1280, mhubert_dim=768, num_heads=8)
            self.encoder_projector = WhisperProjector(downsample_rate=4, idim=3328, odim=3584)
        elif self.cross_type == "whisper_mhubert":
            if self.projector_type == "linear":
                self.encoder_projector = WhisperProjector(downsample_rate=4, idim=2048, odim=3584)
            elif self.projector_type == "qformer":
                # 1. Q-Former
                self.qformer_input_dim = 1280+768  # Whisper的输出维度(1280) + mHuBERT的输出维度(768)
                # 实例化 Q-Former 和可学习的 query_tokens
                self.num_query_tokens = 1 # 这是超参数，可以调整
                self.speech_former, self.speech_query_tokens = BaseModel.build_audio_qformer(1, 2048,2,1)
   
                # 创建一个最终的投影层，将Q-Former的输出(768维)映射到Qwen的输入(3584维)
                qformer_hidden_size = 768 # 这是BERT-base的默认维度
                qwen_hidden_size = self.qwen.config.hidden_size 
                self.final_projector = torch.nn.Linear(qformer_hidden_size, qwen_hidden_size)




    def prompt_wrap(self, speech_embeds):
        batch_size = speech_embeds.size(0)
        prompt_ids = torch.tensor(list(range(20)), dtype=torch.int64, device=speech_embeds.device)
        prompt_ids = prompt_ids.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        prompt_embeds = self.trainable_prompts(prompt_ids)

        wrapped_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        return wrapped_embeds
    def prompt_wrap_with_languages(self, speech_embeds, languages):
        """Wrap speech_embeds with prompts and natural language."""

        device = speech_embeds.device
        prompt_texts = [f"Please transcribe the following audio in {lang}:\n" for lang in languages]

        # 1. Batch tokenize with padding
        prompt_tokens = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False
        ).to(device)

        # 2. Prompt embedding (B, T_prompt, D)
        if self.use_qwen_lora:
            prompt_embeds = self.qwen.model.model.embed_tokens(prompt_tokens.input_ids)
        else:
            prompt_embeds = self.qwen.model.embed_tokens(prompt_tokens.input_ids)
        # 4. 拼接 prompt 和 speech
        wrapped_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)  # (B, T_prompt + T_speech, D)

        return wrapped_embeds
    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def pad_list(self, xs: List[torch.Tensor], pad_value: int):
        """Perform padding for the list of tensors.

        Args:
            xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
            pad_value (float): Value for padding.

        Returns:
            Tensor: Padded tensor (B, Tmax, `*`).

        Examples:
            >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
            >>> x
            [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
            >>> pad_list(x, 0)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 0., 0.],
                    [1., 0., 0., 0.]])

        """
        max_len = max([len(item) for item in xs])
        batchs = len(xs)
        ndim = xs[0].ndim
        if ndim == 1:
            pad_res = torch.zeros(batchs,
                                max_len,
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        elif ndim == 2:
            pad_res = torch.zeros(batchs,
                                max_len,
                                xs[0].shape[1],
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        elif ndim == 3:
            pad_res = torch.zeros(batchs,
                                max_len,
                                xs[0].shape[1],
                                xs[0].shape[2],
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        else:
            raise ValueError(f"Unsupported ndim: {ndim}")
        pad_res.fill_(pad_value)
        for i in range(batchs):
            pad_res[i, :len(xs[i])] = xs[i]
        return pad_res

    def add_eos(self, ys_pad: torch.Tensor, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    
        _eos = torch.tensor([eos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [y for y in ys]
        ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
        return self.pad_list(ys_in, eos), self.pad_list(ys_out, ignore_id)

    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        languages = [key.split('-')[0] for key in batch['keys']]
        B = speech.shape[0]
        D_mel = speech.shape[2]
        batch_waves = []
        # for i in range(B):
        #     batch_waves.append(batch['original_wavs'][i].squeeze(0).numpy())
        # # 1. Prepare inputs for mHuBERT
        

        # 1. Encoder: Directly use the speech input
        assert len(languages) == B
        # --- 1. Prepare inputs and run encoder ---
        # Directly pass speech and its length to the Whisper Encoder without padding
        for i in range(B):
            wav = batch['pcm'][i].squeeze(0).numpy()  # shape: (T,)
            batch_waves.append(wav)  # shape 都是 (480000,)
        assert len(languages) == B
        whisper_features = self.whisper_processor.feature_extractor(
           batch_waves,
           sampling_rate=16000,
            return_tensors="pt",
            padding=False,        # 自动加 pad
            truncation=False      # 自动截断
        ).input_features.to(device)
        encoder_out = self.whisper_encoder(whisper_features)
        encoder_out = encoder_out.last_hidden_state
        #encoder_out, _ = checkpoint.checkpoint(lambda: self.encoder(speech, speech_lengths)) if self.training else self.encoder(speech, speech_lengths)
        #hubert encoder
        #encoder_out, _ = self.encoder(speech, speech_lengths)
        # Prepare inputs for mHuBERT
        batch_waves = []
        for i in range(B):
            batch_waves.append(batch['original_wavs'][i].squeeze(0).numpy())
        hubert_inputs = self.hubert_processor(batch_waves, sampling_rate=16000, return_tensors="pt", padding=True,return_attention_mask=True).to(device)
        # Run mHuBERT encoder
        #hubert_outputs = self.hubert_encoder(hubert_inputs.input_values, attention_mask=hubert_inputs.attention_mask, return_dict=True)
        #hubert_outputs = checkpoint.checkpoint(lambda: self.hubert_encoder(hubert_inputs.input_values, attention_mask=hubert_inputs.attention_mask, return_dict=True)) if self.training else self.hubert_encoder(hubert_inputs.input_values, attention_mask=hubert_inputs.attention_mask, return_dict=True)
        hubert_outputs = self.hubert_encoder(hubert_inputs.input_values, attention_mask=hubert_inputs.attention_mask, return_dict=True)
        encoder_out_hubert = hubert_outputs.last_hidden_state
        encoder_mask = self.hubert_encoder._get_feature_vector_attention_mask(
            encoder_out_hubert.shape[1], hubert_inputs.attention_mask)        
        hubert_attentions = encoder_mask
        if hubert_attentions is None:
            print("None")        
        #hubert_attentions=None
#print(f"speech: {speech.shape}")
        #print(f"hubert_outputs.last_hidden_state: {encoder_out_hubert.shape}")
        #print(f"encoder_out: {encoder_out.shape}")
        if encoder_out.shape[1] > encoder_out_hubert.shape[1]:
            encoder_out = encoder_out[:, :-1, :] 
        elif encoder_out.shape[1] < encoder_out_hubert.shape[1]:
            encoder_out_hubert = encoder_out_hubert[:, :-1, :]
            hubert_attentions = hubert_attentions[:, :-1] 
        assert encoder_out.shape[1] == encoder_out_hubert.shape[1]
        #print(f"speech: {speech.shape}")
        #print(f"hubert_outputs.last_hidden_state: {encoder_out_hubert.shape}")
        #print(f"encoder_out: {encoder_out.shape}")
        #encoder_out = torch.cat([encoder_out, encoder_out_hubert], dim=2)
        # 2. Cross Attention Fusion
        cross_type = self.cross_type
        if cross_type == "cross_attn":
            encoder_out = self.cross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
        elif cross_type == "bicross_attn":
            fused_whisper,fused_mhubert = self.bicross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
            encoder_out = torch.cat([fused_whisper, fused_mhubert], dim=2)  # Concatenate along the feature dimension
        elif cross_type == "gated_cross_attn":
            encoder_out = self.cross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
        elif cross_type == "gated_bicross_attn":
            fused_whisper,fused_mhubert = self.bicross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
            encoder_out = torch.cat([fused_whisper, fused_mhubert], dim=2)
        elif cross_type == "whisper_mhubert":
            encoder_out = torch.cat([encoder_out, encoder_out_hubert], dim=2)
        elif cross_type == "concat_bicross_attn":
            concatenation = self.bicross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
            encoder_out = concatenation
        elif cross_type == "concat_cross_attn":
            concatenation = self.cross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
            encoder_out = concatenation
        elif cross_type == "whisper_mhubert":
            encoder_out = torch.cat([encoder_out, encoder_out_hubert], dim=2)
        if self.projector_type == "linear":
            # 2. WhisperProjector
            speech_embeds = self.encoder_projector(encoder_out)  # Projecting the encoder outputs
        elif self.projector_type == "qformer":
            audio_embeds = encoder_out
            batch, tokens, dim = audio_embeds.shape
            kernel = (1, 17)  # for ced 714ms/per frame (ced 10s: 252 frame), we reduce to about 1.4 frames/second
            audio_embeds_new = F.unfold(
                audio_embeds.transpose(1, 2).unsqueeze(2), kernel_size=kernel, stride=kernel
            )
            audio_embeds_new = audio_embeds_new.view(batch, dim, kernel[1], -1)
            audio_embeds_new = torch.permute(audio_embeds_new, [0, 3, 2, 1])
            audio_embeds = audio_embeds_new.reshape(-1, kernel[1], dim)

            speech_atts = torch.ones(
                audio_embeds.size()[:-1], dtype=torch.long, device=audio_embeds.device
            )
            query_tokens = self.speech_query_tokens.expand(audio_embeds.shape[0], -1, -1)

            audio_embeds = self.speech_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=audio_embeds,
                encoder_attention_mask=speech_atts,
                return_dict=True,
            )["last_hidden_state"]
                # 此处 qformer_output 的形状是 (B * num_windows, num_query_tokens, 768)

                # --- 关键修复：重塑输出，恢复原始批次大小 B ---
                # a. 获取窗口数量
            num_windows = audio_embeds_new.shape[1]
                # b. 将形状变回 (B, num_windows, num_query_tokens, 768)
            qformer_output_reshaped = audio_embeds.view(
                    batch, num_windows, self.num_query_tokens, -1
                )
                # c. 将窗口和query_token维度合并，形成新的序列维度
                #    形状变为 (B, num_windows * num_query_tokens, 768)
            final_qformer_output = qformer_output_reshaped.flatten(start_dim=1, end_dim=2)
            speech_embeds = self.final_projector(final_qformer_output)  # Projecting the Q-Former outputs
        # 3. wrap speech_embeds with prompts
        speech_embeds = self.prompt_wrap_with_languages(speech_embeds, languages)
        
        # 4. prepare inputs for qwen
        to_regress_tokens_in, to_regress_tokens_out = self.add_eos(text, self.qwen.config.eos_token_id, -100)
        to_regress_embeds = self.qwen.model.embed_tokens(to_regress_tokens_in.to(speech.device)) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(to_regress_tokens_in.to(speech.device))

        bos = torch.ones(
            [speech_embeds.shape[0], 1],
            dtype=torch.int64,
            device=to_regress_embeds.device,
        ) * self.qwen.config.bos_token_id  # bos_token_id: 151643
        bos_embeds = self.qwen.model.embed_tokens(bos) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(bos)

        eos = torch.ones(
            [speech_embeds.shape[0], 1],
            dtype=torch.int64,
            device=to_regress_embeds.device,
        ) * self.qwen.config.eos_token_id  # eos_token_id: 151643
        eos_embeds = self.qwen.model.embed_tokens(eos) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(eos)

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds, eos_embeds], dim=1).to(torch.bfloat16)
        
        empty_targets = (
            torch.ones(
                [speech_embeds.shape[0], speech_embeds.shape[1] + 1],
                dtype=torch.long
            ).to(speech.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, to_regress_tokens_out.to(speech.device)], dim=1)
        #targets.requires_grad = False
        # 使用torch.cuda.amp.autocast()进行混合精度训练
       # with torch.cuda.amp.autocast():
            # 7. 运行LLM
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=targets,
        )

            # 8. 清理内存
            #del speech_embeds, inputs_embeds
            #torch.cuda.empty_cache()


        return {"loss": outputs.loss}
        #outputs = self.qwen(
        #    inputs_embeds=inputs_embeds,
        #    return_dict=True,
        #    labels=targets,
        #)
        #loss = outputs.loss
        #return {"loss": loss}
    
    def decode(
        self,
        batch: Dict[str, torch.Tensor],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        qwen_path: str,
    ) -> List[str]:
        """Decode function for inference.

        Args:
            speech (torch.Tensor): Input speech features (B, T, D).
            speech_lengths (torch.Tensor): Lengths of input speech (B, ).
            qwen_path (str): Path to Qwen checkpoint for decoding config.
            batch (dict): Batch dict containing 'keys' for language info.

        Returns:
            List[str]: List of decoded texts.
        """

        device = speech.device
        languages = [key.split('-')[0] for key in batch['keys']]
        B = speech.shape[0]
        speech = speech.to(device)
        #encoder_out, _ = checkpoint.checkpoint(lambda: self.encoder(speech, speech_lengths)) if self.training else self.encoder(speech, speech_lengths)
        # 1. Prepare inputs for mHuBERT
        batch_waves = []
        assert len(languages) == B
        # --- 1. Prepare inputs and run encoder ---
        # Directly pass speech and its length to the Whisper Encoder without padding
        for i in range(B):
            wav = batch['pcm'][i].squeeze(0).numpy()  # shape: (T,)
            batch_waves.append(wav)  # shape 都是 (480000,)
        assert len(languages) == B
        whisper_features = self.whisper_processor.feature_extractor(
           batch_waves,
           sampling_rate=16000,
            return_tensors="pt",
            padding=False,        # 自动加 pad
            truncation=False      # 自动截断
        ).input_features.to(device)
        encoder_out = self.whisper_encoder(whisper_features)
        encoder_out = encoder_out.last_hidden_state
        #encoder_out, _ = checkpoint.checkpoint(lambda: self.encoder(speech, speech_lengths)) if self.training else self.encoder(speech, speech_lengths)
        #hubert encoder
        #encoder_out, _ = self.encoder(speech, speech_lengths)
        # Prepare inputs for mHuBERT
        batch_waves = []
        for i in range(B):
            batch_waves.append(batch['original_wavs'][i].squeeze(0).numpy())
        hubert_inputs = self.hubert_processor(batch_waves, sampling_rate=16000, return_tensors="pt", padding=True,return_attention_mask=True).to(device)
        # Run mHuBERT encoder
        #hubert_outputs = self.hubert_encoder(hubert_inputs.input_values, attention_mask=hubert_inputs.attention_mask, return_dict=True)
        #hubert_outputs = checkpoint.checkpoint(lambda: self.hubert_encoder(hubert_inputs.input_values, attention_mask=hubert_inputs.attention_mask, return_dict=True)) if self.training else self.hubert_encoder(hubert_inputs.input_values, attention_mask=hubert_inputs.attention_mask, return_dict=True)
        hubert_outputs = self.hubert_encoder(hubert_inputs.input_values, attention_mask=hubert_inputs.attention_mask, return_dict=True)
        encoder_out_hubert = hubert_outputs.last_hidden_state
        encoder_mask = self.hubert_encoder._get_feature_vector_attention_mask(
        encoder_out_hubert.shape[1], hubert_inputs.attention_mask)
        hubert_attentions = encoder_mask
#hubert_attentions = hubert_outputs.attentions if hasattr(hubert_outputs, 'attentions') else None
        
        #print(f"speech: {speech.shape}")
        #print(f"hubert_outputs.last_hidden_state: {encoder_out_hubert.shape}")
        #print(f"encoder_out: {encoder_out.shape}")
        if encoder_out.shape[1] > encoder_out_hubert.shape[1]:
            encoder_out = encoder_out[:, :-1, :] 
        elif encoder_out.shape[1] < encoder_out_hubert.shape[1]:
            encoder_out_hubert = encoder_out_hubert[:, :-1, :]
            hubert_attentions = hubert_attentions[:, :-1]
        assert encoder_out.shape[1] == encoder_out_hubert.shape[1]
        #print(f"speech: {speech.shape}")
        #print(f"hubert_outputs.last_hidden_state: {encoder_out_hubert.shape}")
        #print(f"encoder_out: {encoder_out.shape}")
        #encoder_out = torch.cat([encoder_out, encoder_out_hubert], dim=2)
        # 2. Cross Attention Fusion
        #encoder_out = self.cross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
        cross_type = self.cross_type
        if cross_type == "cross_attn":
            encoder_out = self.cross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
        elif cross_type == "bicross_attn":
            fused_whisper,fused_mhubert = self.bicross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
            encoder_out = torch.cat([fused_whisper, fused_mhubert], dim=2)  # Concatenate along the feature dimension
        elif cross_type == "gated_cross_attn":
            encoder_out = self.cross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
        elif cross_type == "gated_bicross_attn":
            fused_whisper,fused_mhubert = self.bicross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
            encoder_out = torch.cat([fused_whisper, fused_mhubert], dim=2)
        elif cross_type == "concat_bicross_attn":
            concatenation = self.bicross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
            encoder_out = concatenation
        elif cross_type == "concat_cross_attn":
            concatenation = self.cross_attn(encoder_out, encoder_out_hubert, whisper_mask=None, mhubert_mask=hubert_attentions)
            encoder_out = concatenation
        elif cross_type == "whisper_mhubert":
            encoder_out=torch.cat([encoder_out, encoder_out_hubert], dim=2)
        # 2. WhisperProjector
        #speech_embeds = self.encoder_projector(encoder_out)  # Projecting the encoder outputs
        if self.projector_type == "linear":
            # 2. WhisperProjector
            speech_embeds = self.encoder_projector(encoder_out)  # Projecting the encoder outputs
        elif self.projector_type == "qformer":
            audio_embeds = encoder_out
            batch, tokens, dim = audio_embeds.shape
            kernel = (1, 17)  # for ced 714ms/per frame (ced 10s: 252 frame), we reduce to about 1.4 frames/second
            audio_embeds_new = F.unfold(
                audio_embeds.transpose(1, 2).unsqueeze(2), kernel_size=kernel, stride=kernel
            )
            audio_embeds_new = audio_embeds_new.view(batch, dim, kernel[1], -1)
            audio_embeds_new = torch.permute(audio_embeds_new, [0, 3, 2, 1])
            audio_embeds = audio_embeds_new.reshape(-1, kernel[1], dim)

            speech_atts = torch.ones(
                audio_embeds.size()[:-1], dtype=torch.long, device=audio_embeds.device
            )
            query_tokens = self.speech_query_tokens.expand(audio_embeds.shape[0], -1, -1)

            audio_embeds = self.speech_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=audio_embeds,
                encoder_attention_mask=speech_atts,
                return_dict=True,
            )["last_hidden_state"]
                # 此处 qformer_output 的形状是 (B * num_windows, num_query_tokens, 768)

                # --- 关键修复：重塑输出，恢复原始批次大小 B ---
                # a. 获取窗口数量
            num_windows = audio_embeds_new.shape[1]
                # b. 将形状变回 (B, num_windows, num_query_tokens, 768)
            qformer_output_reshaped = audio_embeds.view(
                    batch, num_windows, self.num_query_tokens, -1
                )
                # c. 将窗口和query_token维度合并，形成新的序列维度
                #    形状变为 (B, num_windows * num_query_tokens, 768)
            final_qformer_output = qformer_output_reshaped.flatten(start_dim=1, end_dim=2)
            speech_embeds = self.final_projector(final_qformer_output)  # Projecting the Q-Former outputs
        # 3. wrap speech_embeds with prompts
        speech_embeds = self.prompt_wrap_with_languages(speech_embeds, languages)

        # 6. Add BOS
        bos = torch.ones(
            [speech_embeds.shape[0], 1],
            dtype=torch.int64,
            device=device,
        ) * self.qwen.config.bos_token_id
        bos_embeds = self.qwen.model.embed_tokens(bos) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(bos)

        inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1).to(torch.bfloat16)
        
        # 7. Load GenerationConfig
        self.qwen.generation_config = GenerationConfig.from_pretrained(
            qwen_path,
            do_sample=False,
            max_new_tokens=200,
            num_beams=1,
            min_length=1,
            temperature=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0
        )

        # 8. Generate outputs
        outputs = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            generation_config=self.qwen.generation_config
        )

        # 9. Decode tokens to text
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return results
