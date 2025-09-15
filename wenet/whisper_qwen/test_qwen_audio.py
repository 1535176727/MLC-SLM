import torch
import json
import os
import logging
from pprint import pprint
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
qwen_path = "/node/myx/MLC-SLM-LLM/examples/mlcslm/asr/pre_trained_models/Qwen2-Audio-7B"
qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(qwen_path)
print(qwen_model)




def load_whisper_encoder(path: str) -> dict:
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Whisper Checkpoint: loading from whisper checkpoint {}'.format(
        rank, path))
    # import pdb;pdb.set_trace()
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    print(checkpoint)
    checkpoint_encoder = {}
    for k in checkpoint.keys():
        if 'encoder' in k:
            nk = '.'.join(k.split('.')[1:])
            checkpoint_encoder[nk] = checkpoint[k]
    # missing_keys, unexpected_keys = model.encoder.load_state_dict(checkpoint_encoder, strict=False)
    return checkpoint_encoder

whisper_path = "/node/myx/MLC-SLM-LLM/examples/mlcslm/asr/pre_trained_models/whisper-large-v3/wenet_whisper.pt"
whisper_model = torch.load(whisper_path, map_location='cpu', mmap=True)
print(qwen_model)

print(whisper_model.keys())
