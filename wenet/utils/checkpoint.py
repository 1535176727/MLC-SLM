# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
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

import logging
import os
import re

import yaml
import torch
from collections import OrderedDict

import datetime


import logging
import os
import torch
from collections import OrderedDict

def load_projector(model: torch.nn.Module, path: str) -> None:
    """
    从单个检查点文件中，为模型中存在的Projector相关模块（Q-Former或线性层）加载权重。

    Args:
        model (torch.nn.Module): 您的主模型，例如 SHNU_masr 的实例。
        path (str): 包含Projector权重的检查点文件路径。
    """
    rank = int(os.environ.get('RANK', 0))
    logging.info(f'[Rank {rank}] Projector Checkpoint: loading from {path}')
    
    # 1. 加载总的权重字典
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    
    # --- 修复逻辑：分模块检查并加载 ---

    # 2. 尝试加载 encoder_projector (如果存在)
    if hasattr(model, 'encoder_projector'):
        logging.info("Attempting to load weights for 'encoder_projector'...")
        checkpoint_encoder_projector = {}
        for k, v in checkpoint.items():
            if k.startswith('encoder_projector.'):
                nk = k[len('encoder_projector.'):]
                checkpoint_encoder_projector[nk] = v
        
        if checkpoint_encoder_projector:
            model.encoder_projector.load_state_dict(checkpoint_encoder_projector, strict=False)
            logging.info("'encoder_projector' weights loaded.")
        else:
            logging.warning("No weights for 'encoder_projector' found in checkpoint.")
    # 2. 尝试加载 encoder_projector (如果存在)
    if hasattr(model, 'cross_attn'):
        logging.info("Attempting to load weights for 'cross_attn'...")
        checkpoint_cross_attn = {}
        for k, v in checkpoint.items():
            if k.startswith('cross_attn.'):
                nk = k[len('cross_attn.'):]
                checkpoint_cross_attn[nk] = v
        
        if checkpoint_cross_attn:
            model.cross_attn.load_state_dict(checkpoint_cross_attn, strict=False)
            logging.info("'cross_attn' weights loaded.")
        else:
            logging.warning("No weights for 'cross_attn' found in checkpoint.")
    # 2. 尝试加载 encoder_projector (如果存在)
    if hasattr(model, 'bicross_attn'):
        logging.info("Attempting to load weights for 'bicross_attn'...")
        checkpoint_bicross_attn = {}
        for k, v in checkpoint.items():
            if k.startswith('bicross_attn.'):
                nk = k[len('bicross_attn.'):]
                checkpoint_bicross_attn[nk] = v
        
        if checkpoint_bicross_attn:
            model.bicross_attn.load_state_dict(checkpoint_bicross_attn, strict=False)
            logging.info("'bicross_attn' weights loaded.")
        else:
            logging.warning("No weights for 'bicross_attn' found in checkpoint.")
          
    # 3. 尝试加载 speech_former (如果存在)
    if hasattr(model, 'speech_former'):
        logging.info("Attempting to load weights for 'speech_former'...")
        checkpoint_speech_former = {}
        for k, v in checkpoint.items():
            if k.startswith('speech_former.'):
                nk = k[len('speech_former.'):]
                checkpoint_speech_former[nk] = v
        
        if checkpoint_speech_former:
            model.speech_former.load_state_dict(checkpoint_speech_former, strict=False)
            logging.info("'speech_former' weights loaded.")
        else:
            logging.warning("No weights for 'speech_former' found in checkpoint.")

    # 4. 尝试加载 speech_query_tokens (如果存在)
    if hasattr(model, 'speech_query_tokens'):
        logging.info("Attempting to load weights for 'speech_query_tokens'...")
        # 关键修复：直接加载Parameter/Tensor的数据
        if 'speech_query_tokens' in checkpoint:
            # 使用 .data.copy_() 来直接加载张量数据
            model.speech_query_tokens.data.copy_(checkpoint['speech_query_tokens'])
            logging.info("'speech_query_tokens' weights loaded.")
        else:
            logging.warning("No weights for 'speech_query_tokens' found in checkpoint.")

    # 5. 尝试加载 final_projector (如果存在)
    if hasattr(model, 'final_projector'):
        logging.info("Attempting to load weights for 'final_projector'...")
        checkpoint_final_projector = {}
        for k, v in checkpoint.items():
            if k.startswith('final_projector.'):
                nk = k[len('final_projector.'):]
                checkpoint_final_projector[nk] = v

        if checkpoint_final_projector:
            model.final_projector.load_state_dict(checkpoint_final_projector, strict=False)
            logging.info("'final_projector' weights loaded.")
        else:
            logging.warning("No weights for 'final_projector' found in checkpoint.")

def load_whisper_encoder(model: torch.nn.Module, path: str) -> dict:
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Whisper Checkpoint: loading from whisper checkpoint {}'.format(
        rank, path))
    # import pdb;pdb.set_trace()
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    checkpoint_encoder = {}
    for k in checkpoint.keys():
        if 'encoder' in k:
            nk = '.'.join(k.split('.')[1:])
            checkpoint_encoder[nk] = checkpoint[k]
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_encoder, strict=False)
    if rank == 0:
        for key in missing_keys:
            logging.info("missing tensor: {}".format(key))
        for key in unexpected_keys:
            logging.info("unexpected tensor: {}".format(key))
    return model
def load_whisper_lora(model: torch.nn.Module, path: str) -> dict:
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    # 去掉前缀 "whisper_model."，重新组织成新字典
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("whisper_model."):
            new_key = k[len("whisper_model."):]  # 去掉前缀
        else:
            new_key = k
        new_state_dict[new_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict,
                                                            strict=False)
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        logging.info('[Rank {}] Whisper LoRA Checkpoint: loading from whisper lora checkpoint {}'.format(
            rank, path))
        for key in missing_keys:
            logging.info("missing tensor: {}".format(key))
        for key in unexpected_keys:
            logging.info("unexpected tensor: {}".format(key))
    return model
def load_mhubert_ckpt(model: torch.nn.Module, path: str) -> dict:
    #print(model)
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        logging.info('[Rank {}] HuBERT Checkpoint: loading from hubert checkpoint {}'.format(
        rank, path))
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    source_state_dict = checkpoint

    # 确认权重在 'model' key下
    if 'model' in source_state_dict:
        source_state_dict = source_state_dict['model']


    # --- 2. 创建新的state_dict并使用更全面的规则进行映射 ---

    new_state_dict = {}

    for old_key, value in source_state_dict.items():
        new_key = old_key
        if new_key == "encoder.encoders.layer_norm.weight":
            new_key = new_key.replace("encoder.encoders.layer_norm.weight", "feature_projection.layer_norm.weight")
        if new_key == "encoder.encoders.layer_norm.bias":
            new_key = new_key.replace("encoder.encoders.layer_norm.bias", "feature_projection.layer_norm.bias")
        # 规则A：处理最外层的前缀
        if new_key.startswith("encoder.encoders."):
            new_key = new_key.replace("encoder.encoders.", "")
        
        # 规则B: 修正 Transformer 内部层的命名
        # self_attn -> attentioi
        new_key = new_key.replace("self_attn", "attention")
        # fc1 -> feed_forward.intermediate_dense
        new_key = new_key.replace("fc1", "feed_forward.intermediate_dense")
        # fc2 -> feed_forward.output_dense
        new_key = new_key.replace("fc2", "feed_forward.output_dense")
        # 第一个 layer norm
        new_key = new_key.replace("attention_layer_norm", "layer_norm") # 注意：这里可能需要根据您的模型微调，如果self_attn_layer_norm更准确则使用它
        if "attention.output.dense" not in new_key: # 避免错误替换 final_layer_norm 中的 'layer_norm'
            new_key = new_key.replace("self_attn_layer_norm", "layer_norm")


        # 规则C: 修正 Feature Extractor 的命名
        # 使用正则表达式来匹配 conv_layers.[数字]
        if "feature_extractor.conv_layers" in new_key:
            # e.g., conv_layers.0.0.weight -> conv_layers.0.conv.weight
            new_key = re.sub(r'\.conv_layers\.(\d+)\.0\.', r'.conv_layers.\1.conv.', new_key)
            # e.g., conv_layers.0.2.weight -> conv_layers.0.layer_norm.weight
            new_key = re.sub(r'\.conv_layers\.(\d+)\.2\.', r'.conv_layers.\1.layer_norm.', new_key)

        # 规则D: 修正其他模块的命名
        if "post_extract_proj" in new_key:
            new_key = new_key.replace("post_extract_proj", "feature_projection.projection")
        #encoder.encoders.layer_norm.weight
        #encoder.encoders.layer_norm.bias

        if "encoder.pos_conv.0" in new_key:
            new_key = new_key.replace("encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.parametrizations.weight.original0")
            new_key = new_key.replace("encoder.pos_conv.0.weight_v", "encoder.pos_conv_embed.conv.parametrizations.weight.original1")
            new_key = new_key.replace("encoder.pos_conv.0.bias", "encoder.pos_conv_embed.conv.bias")
            
        if new_key == "mask_emb":
            new_key = "masked_spec_embed"


        # 只有在转换后的key确实存在于目标模型中时，才添加到新字典
        if new_key in model.state_dict():
            new_state_dict[new_key] = value
    print(f"Remapping complete. {len(new_state_dict)} parameters were successfully mapped.")
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if rank == 0:
        for key in missing_keys:
            logging.info("missing tensor: {}".format(key))
        for key in unexpected_keys:
            logging.info("unexpected tensor: {}".format(key))
    return model

def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Checkpoint: loading from checkpoint {}'.format(
        rank, path))
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)
    if rank == 0:
        for key in missing_keys:
            logging.info("missing tensor: {}".format(key))
        for key in unexpected_keys:
            logging.info("unexpected tensor: {}".format(key))
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs
def load_encoder(model: torch.nn.Module, path: str) -> None:
    """
    从【单个】检查点文件中为两个编码器（whisper_encoder, hubert_encoder）加载权重。
    这个函数保持了你原有的“遍历一次权重”的核心逻辑。

    Args:
        model (nn.Module): 你的主模型，例如 SHNU_masr 的实例。
        path (str): 包含两个编码器权重的【单个】检查点文件路径。
    """
    rank = int(os.environ.get('RANK', 0))
    logging.info(f'[Rank {rank}] Dual-Encoder Checkpoint: loading from single checkpoint {path}')
    
    # 从单一文件加载总的权重字典
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    
    # 准备两个独立的字典，分别存放两个编码器的权重
    checkpoint_whisper_encoder = {}
    checkpoint_hubert_encoder = {}

    # 遍历检查点中的每一个权重键 (这是你原有逻辑的核心)
    for k, v in checkpoint.items():
        # 判断当前权重属于哪个编码器，并进行处理
        if k.startswith('whisper_encoder.'):
            # 如果键以 "whisper_encoder." 开头，则处理后放入 whisper 字典
            # 'whisper_encoder.conv1.weight' -> 'conv1.weight'
            nk = k[len('whisper_encoder.'):]
            checkpoint_whisper_encoder[nk] = v
        elif k.startswith('hubert_encoder.'):
            # 如果键以 "hubert_encoder." 开头，则处理后放入 hubert 字典
            # 'hubert_encoder.encoder.layer.0.weight' -> 'encoder.layer.0.weight'
            nk = k[len('hubert_encoder.'):]
            checkpoint_hubert_encoder[nk] = v
            
    # 分别加载权重到对应的模块
    logging.info("Loading weights into model.whisper_encoder...")
    missing_keys_w, unexpected_keys_w = model.whisper_encoder.load_state_dict(checkpoint_whisper_encoder, strict=False)
    
    logging.info("Loading weights into model.hubert_encoder...")
    missing_keys_h, unexpected_keys_h = model.hubert_encoder.load_state_dict(checkpoint_hubert_encoder, strict=False)

    # 统一打印加载信息
    if rank == 0:
        if missing_keys_w:
            logging.warning("Whisper Encoder - Missing Tensors: {}".format(missing_keys_w))
        if unexpected_keys_w:
            logging.warning("Whisper Encoder - Unexpected Tensors: {}".format(unexpected_keys_w))
        if missing_keys_h:
            logging.warning("Hubert Encoder - Missing Tensors: {}".format(missing_keys_h))
        if unexpected_keys_h:
            logging.warning("Hubert Encoder - Unexpected Tensors: {}".format(unexpected_keys_h))
        
    logging.info(f'[Rank {rank}] Successfully processed dual-encoder loading.')

def save_state_dict_and_infos(state_dict, path: str, infos=None):
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(
        rank, path))
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    print("save checkpoint,line 118")
    print(state_dict)
    save_state_dict_and_infos(state_dict, path, infos)


def filter_modules(model_state_dict, modules):
    rank = int(os.environ.get('RANK', 0))
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods and rank == 0:
        logging.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                     enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location='cpu')
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        logging.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs
