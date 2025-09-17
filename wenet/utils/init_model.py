# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
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

import os
import logging as logger
import torch
from transformers import Qwen2AudioForConditionalGeneration,AutoProcessor
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.ctl_model.asr_model_ctl import CTLModel
from wenet.ctl_model.encoder import (DualConformerEncoder,
                                     DualTransformerEncoder)
from wenet.e_branchformer.encoder import EBranchformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder
from wenet.finetune.lora.utils import (inject_lora_to_model,
                                       mark_only_lora_as_trainable)
from wenet.firered.encoder import FireRedConformerEncoder
from wenet.firered.model import FireReadModel
from wenet.k2.model import K2Model
from wenet.paraformer.cif import Cif
from wenet.paraformer.layers import SanmDecoder, SanmEncoder
from wenet.paraformer.paraformer import Paraformer, Predictor
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.ssl.init_model import WENET_SSL_MODEL_CLASS
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.utils.checkpoint import load_checkpoint, load_trained_modules, load_projector, load_whisper_encoder,load_whisper_lora,load_mhubert_ckpt,load_encoder
from wenet.utils.cmvn import load_cmvn
from wenet.whisper.whisper import Whisper
from wenet.whisper.whisper_finetuning import WhisperFinetuning

from wenet.SHNU_masr.SHNU_masr import SHNU_masr

#from wenet.SHNU_masr.SHNU_masr_qformer import SHNU_masr
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import WhisperModel
#使用mhubert作为encoder
from transformers import HubertModel, Wav2Vec2FeatureExtractor
WENET_ENCODER_CLASSES = {
    "transformer": TransformerEncoder,
    "conformer": ConformerEncoder,
    "squeezeformer": SqueezeformerEncoder,
    "efficientConformer": EfficientConformerEncoder,
    "branchformer": BranchformerEncoder,
    "e_branchformer": EBranchformerEncoder,
    "dual_transformer": DualTransformerEncoder,
    "dual_conformer": DualConformerEncoder,
    'sanm_encoder': SanmEncoder,
    "firered_conformer": FireRedConformerEncoder,
}

WENET_DECODER_CLASSES = {
    "transformer": TransformerDecoder,
    "bitransformer": BiTransformerDecoder,
    "sanm_decoder": SanmDecoder,
}

WENET_CTC_CLASSES = {
    "ctc": CTC,
}

WENET_PREDICTOR_CLASSES = {
    "rnn": RNNPredictor,
    "embedding": EmbeddingPredictor,
    "conv": ConvPredictor,
    "cif_predictor": Cif,
    "paraformer_predictor": Predictor,
}

WENET_JOINT_CLASSES = {
    "transducer_joint": TransducerJoint,
}

WENET_MODEL_CLASSES = {
    "asr_model": ASRModel,
    "ctl_model": CTLModel,
    "whisper": WhisperFinetuning,
    "SHNU-mASR": SHNU_masr,
    "whisper_finetuning": WhisperFinetuning,
    "firered": FireReadModel,
    "k2_model": K2Model,
    "transducer": Transducer,
    'paraformer': Paraformer,
}


def init_speech_model(args, configs, tokenizer, inference_mode):
    # TODO(xcsong): Forcefully read the 'cmvn' attribute.
    if configs.get('cmvn', None) == 'global_cmvn':
        mean, istd = load_cmvn(configs['cmvn_conf']['cmvn_file'],
                               configs['cmvn_conf']['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = "whisper_mhubert"
    decoder_type = configs["decoder"]
    ctc_type = configs.get('ctc', 'ctc')
    training = True
    if inference_mode:
        training = False
    if encoder_type == "whisper_mhubert":
        whisper = WhisperForConditionalGeneration.from_pretrained(
            configs['whisper_path'],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            apply_spec_augment=True,
            mask_feature_prob=0.2,
            mask_feature_min_masks=2,
            mask_feature_length=10,
            mask_time_prob=0.2,
            mask_time_min_masks=2,
            mask_time_length=50,
        )
        whisper.config.training = training
        if configs["whisper_encoder_lora"]:
            logger.info("Using Whisper encoder with LoRA")
            target_modules = ["q_proj", "v_proj"]
            peft_config = LoraConfig(
                r = configs['whisper_encoder_lora_rank'],
                lora_alpha = configs['whisper_encoder_lora_alpha'],
                lora_dropout = configs['whisper_encoder_lora_dropout'],
                bias = "none",
                target_modules=target_modules,
            )
            whisper = get_peft_model(whisper, peft_config)
            #print("model after lora:", whisper)
            if configs.get('whisper_lora_ckpt', None) is not None:
                logger.info(f"Loading Whisper encoder LoRA from {configs['whisper_lora_ckpt']}")
                whisper = load_whisper_lora(whisper, configs['whisper_lora_ckpt'])
            else:
                logger.info(f"Loading Whisper encoder from {configs['whisper_ckpt']}")
                whisper = load_whisper_encoder(whisper, configs["whisper_ckpt"])
            whisper_model = whisper
            print("whisper_model:", whisper_model)
        else:
            logger.info(f"Loading Whisper encoder from {configs['whisper_ckpt']}")
            whisper = load_whisper_encoder(whisper, configs["whisper_ckpt"])
            whisper_model = whisper
            print("whisper_model:", whisper_model)
        whisper_encoder = whisper_model
        whisper_processor = WhisperProcessor.from_pretrained(
            configs['whisper_path'],
            local_files_only=True,
        )
        logger.info("use_mhubert: True")
        logger.info(f"Loading HuBERT from {configs['mhubert_path']}")
        hubert_encoder = HubertModel.from_pretrained(
            configs['mhubert_path'],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        if configs.get('mhubert_ckpt', None) is not None:
            logger.info(f"Loading HuBERT encoder from {configs['mhubert_ckpt']}")
            hubert_encoder = load_mhubert_ckpt(hubert_encoder, configs['mhubert_ckpt'])
        hubert_encoder.config.training = training
        hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(configs['mhubert_path'], local_files_only=True)

    else:
        raise NotImplementedError(f"Encoder type {encoder_type} is not implemented.")

    
    qwen2audioencoder = None
    if hasattr(configs, 'use_qwen2audioencoder') and configs['use_qwen2audioencoder']:
        logger.info("use_qwen2audioencoder: True")
        logger.info(f"Loading Qwen2AudioEncoder from {configs['qwen2audioencoder_path']}")
        qwen2audioencoder = Qwen2AudioForConditionalGeneration.from_pretrained(
            configs['qwen2audioencoder_path'],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
 #       qwen2audioprocessor = AutoProcessor.from_pretrained(configs['qwen2audioencoder_path'])
        qwen2audioencoder = qwen2audioencoder.audio_tower
    else:
        qwen2audioencoder = None
   #  hubert_encoder = None
    if hasattr(configs, 'use_mhubert') and configs['use_mhubert']:
        logger.info("use_mhubert: True")
        logger.info(f"Loading HuBERT from {configs['mhubert_path']}")
        hubert_encoder = HubertModel.from_pretrained(
            configs['mhubert_path'],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(configs['mhubert_path'], local_files_only=True)

    if decoder_type == "qwen":
        qwen = AutoModelForCausalLM.from_pretrained(
            configs['qwen_path'],
            attn_implementation=configs['attn_implementation'],
            torch_dtype=torch.bfloat16,
        )
        if configs['use_qwen_lora']:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj","gate_proj", "down_proj"]
            #target_modules = ["q_proj","v_proj"]
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=inference_mode, 
                r=configs['qwen_lora_rank'], 
                lora_alpha=configs['qwen_lora_alpha'], 
                lora_dropout=configs['qwen_lora_dropout'],
                target_modules=target_modules,
            )
            qwen = get_peft_model(qwen, peft_config)
    elif decoder_type == "whisper_decoder" and encoder_type == "whisper_encoder":
        if configs["whisper_decoder_lora"]:
            logger.info("Using Whisper decoder with LoRA")
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj","gate_proj", "down_proj"]
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, 
                inference_mode=inference_mode, 
                r=configs['whisper_decoder_lora_rank'], 
                lora_alpha=configs['whisper_decoder_lora_alpha'], 
                lora_dropout=configs['whisper_decoder_lora_dropout'],
                target_modules=target_modules,
            )
            whisper = get_peft_model(whisper, peft_config)
        whisper_model = whisper

    else:
        decoder = WENET_DECODER_CLASSES[decoder_type](vocab_size,
                                                  encoder.output_size(),
                                                  **configs['decoder_conf'])

    ctc = None
    model_type = configs.get('model', 'asr_model')
    if model_type == "SHNU-mASR":
        model = WENET_MODEL_CLASSES[model_type](
            whisper_encoder=(
                whisper_model.base_model.model.model.encoder
                if configs["whisper_encoder_lora"]
                else whisper_model.model.encoder
            ),
            whisper_processor=whisper_processor,
            hubert_encoder=hubert_encoder,
            hubert_processor=hubert_processor,
            qwen=qwen,
            tokenizer=tokenizer,
            use_qwen_lora=configs['use_qwen_lora'],
        )
    elif model_type == "whisper":
        model = WENET_MODEL_CLASSES[model_type](
            whisper_model=whisper_model,
            tokenizer=tokenizer,
            whisper_encoder_lora=configs.get('whisper_encoder_lora', False),
            whisper_decoder_lora=configs.get('whisper_decoder_lora', False),
            whisper_processor=whisper_processor,
        )
    elif model_type == "transducer":
        predictor_type = configs.get('predictor', 'rnn')
        joint_type = configs.get('joint', 'transducer_joint')
        predictor = WENET_PREDICTOR_CLASSES[predictor_type](
            vocab_size, **configs['predictor_conf'])
        joint = WENET_JOINT_CLASSES[joint_type](vocab_size,
                                                **configs['joint_conf'])
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            blank=0,
            predictor=predictor,
            encoder=encoder,
            attention_decoder=decoder,
            joint=joint,
            ctc=ctc,
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
            **configs['model_conf'])
    elif model_type == 'paraformer':
        predictor_type = configs.get('predictor', 'cif')
        predictor = WENET_PREDICTOR_CLASSES[predictor_type](
            **configs['predictor_conf'])
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            ctc=ctc,
            **configs['model_conf'],
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
        )
    elif model_type in WENET_SSL_MODEL_CLASS.keys():
        from wenet.ssl.init_model import init_model as init_ssl_model
        model = init_ssl_model(configs, encoder)
    else:
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
            **configs['model_conf'])
    return model, configs


def init_model(args, configs, tokenizer, inference_mode=False):

    model_type = configs.get('model', 'asr_model')
    configs['model'] = model_type
    model, configs = init_speech_model(args, configs, tokenizer, inference_mode)
    if model_type == 'SHNU-mASR' and configs['projector_checkpoint'] is not None:
        load_projector(model, configs['projector_checkpoint'])
        #load_encoder(model,configs['encoder_checkpoint'])
    # If specify checkpoint, load some info from checkpoint
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        print(args.checkpoint)
        infos = load_checkpoint(model, args.checkpoint)
    elif hasattr(args, 'enc_init') and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    print("infos:",infos)
    configs["init_infos"] = infos

    if hasattr(args, 'use_lora') and args.use_lora:
        if hasattr(args, 'lora_ckpt_path') and args.lora_ckpt_path:
            load_checkpoint(model, args.lora_ckpt_path)
    if configs.get("use_qwen"):
        if configs.get("qwen_ckpt_path"):
            load_checkpoint(model, configs["qwen_ckpt_path"])
    # Trye to tie some weights
    # if hasattr(model, 'tie_or_clone_weights'):
    #     if not hasattr(args, 'jit'):
    #         jit = True  # i.e. export onnx/jit/ipex
    #     else:
    #         jit = False
    #     model.tie_or_clone_weights(jit)

    if hasattr(args, 'only_optimize_lora') and args.only_optimize_lora:
        mark_only_lora_as_trainable(model, bias='lora_only')

    if int(os.environ.get('RANK', 0)) == 0:
        print(configs)
        print(model)
    return model, configs
