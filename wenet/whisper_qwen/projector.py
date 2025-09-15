import torch
import torch.nn as nn
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """多头注意力层"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size必须能被num_heads整除"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 投影并分离多头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, v)
        
        # 重组多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(context)

class FeedForward(nn.Module):
    """前馈网络层"""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerProjector(nn.Module):
    """将Whisper + Qwen-Audio编码器输出投影到Qwen隐藏维度"""

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float = 0.1,
        num_layers: int = 2,
        num_heads: int = 8,
        intermediate_size: int = 2048
    ):
        super().__init__()
        self.input_layer = nn.Linear(idim, 512)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.odim = odim
        
        # 使用torch.nn.TransformerEncoder，更高效的实现
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,  # 更稳定的Pre-LN架构
            dtype=torch.float16  # 显式指定数据类型
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(512,odim)
        # 初始化梯度检查点标志
        self._gradient_checkpointing = False
        
        # 将所有参数转换为float16
        self.to(torch.float16)

    def gradient_checkpointing_enable(self):
        """启用梯度检查点"""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        self._gradient_checkpointing = False

    def output_size(self) -> int:
        return self.odim

    def forward(
        self,
        xs: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入序列。

        Args:
            xs (torch.Tensor): 输入张量 (#batch, time, idim)
            masks (torch.Tensor): 掩码张量 (#batch, time)

        Returns:
            torch.Tensor: 输出张量 (#batch, time, attention_dim)
            torch.Tensor: 掩码张量 (#batch, time)
        """
        # 确保输入是float16类型
        if xs.dtype != torch.float16:
            xs = xs.to(torch.float16)
        
        # 1. 输入投影
        x = self.input_layer(xs)
        x = self.dropout(x)
        
        # 确保x需要梯度
        if not x.requires_grad and self.training:
            x.requires_grad_(True)

        # 3. 通过Transformer层
        if self._gradient_checkpointing and self.training:
            def custom_forward(src, key_padding_mask):
                return self.transformer(
                    src,
                    src_key_padding_mask=key_padding_mask
                )
            
            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                x,
                masks,
                use_reentrant=False,
                preserve_rng_state=False  # 提高性能
            )
        else:
            output = self.transformer(x, src_key_padding_mask=masks)
        output = self.output_layer(output)
        return output, masks 

 
