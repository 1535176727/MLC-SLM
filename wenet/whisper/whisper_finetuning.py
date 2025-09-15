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
class WhisperFinetuning(torch.nn.Module):

    def __init__(
        self,
        whisper_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        whisper_encoder_lora: bool,
        whisper_decoder_lora: bool,
        whisper_processor: AutoProcessor,
    ):
        super().__init__()
        self.whisper_model = whisper_model
        self.tokenizer = tokenizer
        self.whisper_encoder_lora = whisper_encoder_lora
        self.whisper_decoder_lora = whisper_decoder_lora
        self.whisper_processor = whisper_processor



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

            # # 手动填充或截断
            # if len(wav) < target_length:
            #     pad_width = target_length - len(wav)
            #     wav = numpy.pad(wav, (0, pad_width), mode='constant')  # 后端补零
            # else:
            #     wav = wav[:target_length]  # 超过部分直接截断

            batch_waves.append(wav)  # shape 都是 (480000,)
        #print("text id:",  text)
        #print("text id:",  text)
        #print("text:",  self.tokenizer.batch_decode(text, skip_special_tokens=False))
        decoder_input_ids = text[:, :-1]  # 去掉 <eos>
        #print("decoder_input_ids:", decoder_input_ids)
        #print("decoder_input_ids:", decoder_input_ids)
        
        decoder_labels = text[:, 1:]     # 对齐目标是下一个 token
        #print("decoder_labels:", decoder_labels)
        #print("decoder_labels:", decoder_labels)
        input_features = self.whisper_processor.feature_extractor(
            batch_waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,        # 不再做padding
            truncation=True      # 也不做截断
        ).input_features.to(device)

        whisper_outputs = self.whisper_model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        # 模型的预测 logits
        logits = whisper_outputs.logits  # shape: (batch_size, seq_len, vocab_size)

        # 取 argmax 得到预测 token ids
        pred_ids = torch.argmax(logits, dim=-1)  # shape: (batch_size, seq_len)

        # labels 是传进模型的 decoder_input_ids 的下一个 token，通常在外部准备好的
        label_ids = decoder_labels
        #print("pred_ids:", pred_ids)
        #print("label_ids:", label_ids)
        # 反编码为字符串
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        #print("pred_str:", pred_str)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        #print("label_str:", label_str)
        P = self.tokenizer.batch_decode(pred_ids)
        L = self.tokenizer.batch_decode(label_ids)
        #print("P:", P)
        #print("L:", L)
        log_file = "log.txt"
        metric = jiwer.wer(label_str,pred_str)
        wer = 100 * metric
        with open(log_file, "a", encoding="utf-8") as f:
            for i in range(len(P)):
                f.write(f"Example {i}:\n")
                f.write(f"text: {text[i]}\n")
                f.write(f"decoder_input_ids: {decoder_input_ids[i]}\n")
                f.write(f"decoder_labels: {decoder_labels[i]}\n")
                f.write(f"P: {P[i]}\n")
                f.write(f"L: {L[i]}\n")
                f.write(f"Label: {label_str[i]}\n")
                f.write(f"Predicted: {pred_str[i]}\n\n")
                f.write(f"Batch WER: {wer:.2f}%\n")
        logits = whisper_outputs.logits
        #print("logits:", logits.shape)
        # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # loss = loss_fct(logits.reshape(-1, logits.size(-1)), decoder_labels.reshape(-1))
        labels = decoder_labels  # shape: (B, T)

        # 只忽略真正的 padding，而不是第一个 <|endoftext|>
        # 判断每个位置是否是 padding
        loss_mask = torch.ones_like(labels, dtype=torch.bool)
        # 假设最多允许一个 endoftext 是有效的，其余的是 padding
        for i in range(labels.size(0)):  # batch-wise处理
            # 找到第一个 <|endoftext|> 的索引
            end_indices = (labels[i] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(end_indices) > 0:
                # 第一个保留，后面都是 padding
                loss_mask[i, end_indices[1:]] = False

        # 现在构建 loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss = (loss * loss_mask.reshape(-1)).sum() / loss_mask.sum()
        return {"loss": loss}

    def decode(
        self,
        batch: Dict[str, torch.Tensor],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> List[str]:
        device = speech.device
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

            # 手动填充或截断
            if len(wav) < target_length:
                pad_width = target_length - len(wav)
                wav = numpy.pad(wav, (0, pad_width), mode='constant')  # 后端补零
            else:
                wav = wav[:target_length]  # 超过部分直接截断

            batch_waves.append(wav)  # shape 都是 (480000,)

        # 使用 processor 提取特征（自动加 pad）
        input_features = self.whisper_processor.feature_extractor(
            batch_waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,        # 不再做padding
            truncation=False      # 也不做截断
        ).input_features.to(device)
        # 构造 decoder_start_token（手动传入语言 token）
        # 默认使用 batch 中的语言
        text = batch['target'].to(device)
        decoder_input_ids = text[:, :-1]
        
        # 生成
        generated_ids = self.whisper_model.generate(
            input_features=input_features,
            max_new_tokens=200,     # 控制生成长度
            do_sample=False         # greedy decode
        )
        print("generated_ids:", generated_ids.shape)
        # 解码成文本
        predicted_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("predicted_texts:",  self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
        
        return predicted_texts

