from __future__ import annotations

from typing import TYPE_CHECKING

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
    RMSNorm,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import init_logger, nvtx_annotate

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        match config.hidden_act:
            case "silu":
                self.act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation function: {act_fn}")

        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class RopeAttn(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _log(stage: str, t: torch.Tensor):
            if self.layer_id in [0, 1]:
                logger.info_rank0(f"Layer {self.layer_id} RopeAttn {stage}: shape={t.shape}")

        _log("input", x)
        qkv = self.qkv_proj.forward(x)
        del x
        _log("after qkv_proj", qkv)
        o = self.attn.forward(qkv)
        _log("after attn", o)
        out = self.o_proj.forward(o)
        _log("after o_proj", out)
        return out


__all__ = ["GatedMLP", "RopeAttn"]
