# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2025 ASLP@NPU for MLC-SLM Baseline. (authors: Bingshen Mu)
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
from typing import Dict, List, Optional, Tuple
import torch
import torch.utils.checkpoint as checkpoint
from wenet.transformer.encoder import BaseEncoder
import numpy
from wenet.transformer.search import DecodeResult
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, Qwen2AudioForConditionalGeneration
from wenet.transformer.subsampling import WhisperProjector
from transformers import AutoProcessor
import jiwer
class WhisperQwen(torch.nn.Module):

    def __init__(
        self,
        encoder: torch.nn.Module,
        tokenizer: AutoTokenizer,
        use_qwen_lora: bool,
        qwen: torch.nn.Module,
        whisper_processor: AutoProcessor,
    ):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.whisper_processor = whisper_processor
        self.use_qwen_lora = use_qwen_lora
        self.qwen = qwen
        self.encoder_projector = WhisperProjector(downsample_rate=4, idim=1280, odim=3584)



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
        text_lengths = batch['target_lengths'].to(device)
        languages = [key.split('-')[0] for key in batch['keys']]
        
        B = speech.shape[0]
        D_mel = speech.shape[2]
        target_length = 16000 * 30  # 480000 samples for 30 seconds at 16kHz
        batch_waves = []

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
        encoder_out = self.encoder(whisper_features)

        speech_embeds = self.encoder_projector(encoder_out.last_hidden_state)
        speech_embeds = self.prompt_wrap_with_languages(speech_embeds, languages)
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
        # 7. 运行LLM
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=targets,
        )

        return {"loss": outputs.loss}

    def decode(
        self,
        batch: Dict[str, torch.Tensor],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        qwen_path: str,
    ) -> List[str]:
        #device = torch.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        languages = [key.split('-')[0] for key in batch['keys']]
        
        B = speech.shape[0]
        D_mel = speech.shape[2]
        target_length = 16000 * 30  # 480000 samples for 30 seconds at 16kHz
        batch_waves = []

        for i in range(B):
            wav = batch['pcm'][i].squeeze(0).numpy()  # shape: (T,)
            batch_waves.append(wav)  # shape 都是 (480000,)

        # 使用 processor 提取特征（自动加 pad）
        input_features = self.whisper_processor.feature_extractor(
            batch_waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,        # 不再做padding
            truncation=False      # 也不做截断
        ).input_features.to(device)
        encoder_out = self.encoder(input_features)
        speech_embeds = self.encoder_projector(encoder_out.last_hidden_state)
        speech_embeds = self.prompt_wrap_with_languages(speech_embeds, languages)
        bos = torch.ones(
            [speech_embeds.shape[0], 1],
            dtype=torch.int64,
            device=device,
        ) * self.qwen.config.bos_token_id
        bos_embeds = self.qwen.model.embed_tokens(bos) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(bos)

        inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1).to(torch.bfloat16)
        self.qwen.generation_config = GenerationConfig.from_pretrained(
            qwen_path,
            do_sample=False,
            max_new_tokens=200,
            num_beams=1,
            min_length=1,
            temperature=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 8. Generate outputs
        outputs = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            generation_config=self.qwen.generation_config
        )

        # 9. Decode tokens to text
        #print(self.tokenizer.batch_decode(outputs, skip_special_tokens=False))
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return results

