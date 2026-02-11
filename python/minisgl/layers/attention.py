from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = divide_even(num_qo_heads, tp_size)
        self.num_kv_heads = divide_even(num_kv_heads, tp_size)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        # 获取当前的全局上下文
        ctx = get_global_ctx()
        # 从上下文的 batch 中获取注意力元数据
        metadata = ctx.batch.attn_metadata
        # 将输入张量 qkv 切分为 query, key, value 张量
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        # 如果定义了 q_norm，则对 query 张量进行原地归一化
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        # 如果定义了 k_norm，则对 key 张量进行原地归一化
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # 如果启用了 rotary (RoPE)，则对 query 和 key 应用旋转位置编码
        if self.rotary:
            q, k = self.rotary.forward(metadata.positions, q, k)
        # 将 query 张量重塑为 (batch_size * seq_len, num_heads, head_dim)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        # 使用注意力后端的 forward 方法计算注意力输出
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        # 将输出重塑回 (batch_size * seq_len, hidden_dim) 并返回
        return o.view(-1, self.qo_attn_dim)
