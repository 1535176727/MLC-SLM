import abc
import logging
import re

import torch
import torch.nn as nn
from peft import get_peft_model

from wenet.SHNU_masr.Qformer import BertConfig, BertLMHeadModel

class BaseModel(abc.ABC, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

    @staticmethod
    def build_audio_qformer(
        num_query_token: int,
        audio_width: int,
        num_hidden_layers: int = 2,
        cross_attention_freq: int = 1
    ) -> tuple[nn.Module, nn.Parameter]:

        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        
        encoder_config.encoder_width = audio_width
        
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        qformer = BertLMHeadModel(config=encoder_config)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        qformer.cls = None
        qformer.bert.embeddings.word_embeddings = None
        qformer.bert.embeddings.position_embeddings = None
        for layer in qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
            
        return qformer, query_tokens