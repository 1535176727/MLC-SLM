import abc
import logging
import re

import torch
import torch.nn as nn
from peft import get_peft_model

from wenet.SHNU_masr.Qformer import BertConfig, BertLMHeadModel


class BaseModel(abc.ABC, nn.Module):
    """
    一个抽象基类，提供了构建Q-Former的通用工具方法。
    """
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
        """
        构建并初始化一个Q-Former模型及其可学习的查询向量 (Query Tokens)。
        这是一个静态方法，因为它不依赖于任何类实例(self)的状态。

        Args:
            num_query_token (int): 查询向量的数量。
            audio_width (int): 输入的音频/视觉特征维度。
            num_hidden_layers (int, optional): Q-Former的层数。默认为 2。
            cross_attention_freq (int, optional): 交叉注意力的频率。默认为 1。

        Returns:
            tuple[nn.Module, nn.Parameter]: 返回Q-Former模型和查询向量。
        """
        # 1. 配置 Q-Former
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        
        # 关键: 将传入的 audio_width 赋值给 encoder_config.encoder_width
        # 这可以确保交叉注意力层被正确初始化，以接受正确维度的输入
        encoder_config.encoder_width = audio_width
        
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        # 2. 创建 Q-Former 模型
        qformer = BertLMHeadModel(config=encoder_config)

        # 3. 创建可学习的查询向量
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        # 4. 清理不需要的模块 (针对特定用法进行优化)
        qformer.cls = None
        qformer.bert.embeddings.word_embeddings = None
        qformer.bert.embeddings.position_embeddings = None
        for layer in qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
            
        return qformer, query_tokens