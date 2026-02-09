from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata
from minisgl.core import Batch


@dataclass
class TorchMetadata(BaseAttnMetadata):
    batch: Batch | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # We need to return the index of the last token for each request in the current batch.
        # 'positions' has shape [total_batch_tokens].
        # We can compute it from batch.reqs.
        if self.batch is None:
            raise ValueError("Batch is not attached to metadata")

        reqs = self.batch.reqs
        ends = []
        curr = 0
        for req in reqs:
            curr += req.extend_len
            ends.append(curr - 1)
        return torch.tensor(ends, device=self.positions.device, dtype=torch.long)


class TorchAttentionBackend(BaseAttnBackend):
    def __init__(self, config, kvcache, page_table):
        self.config = config
        self.kvcache = kvcache
        self.page_table = page_table
        self.device = kvcache.device

    def prepare_metadata(self, batch: Batch) -> None:
        # Create a metadata object.
        # For Torch backend, we just need positions for Rotary Embedding (calculated in layers)
        # We need to set batch.attn_metadata.

        # We need 'positions' for Rotary.
        # utils.make_positions helps.
        from minisgl.attention.utils import make_positions

        positions = make_positions(self.device, batch.reqs)

        batch.attn_metadata = TorchMetadata(positions=positions, batch=batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        pass

    def prepare_for_capture(self, batch: Batch) -> None:
        pass

    def prepare_for_replay(self, batch: Batch) -> None:
        pass

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # Store KV
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        start_idx = 0
        outputs = []

        # Access cache buffers
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)

        # Reshape to [num_tokens, num_heads, head_dim] if not already
        storage_shape = self.kvcache._storage_shape
        # storage_shape is (num_pages, local_kv_heads, head_dim)
        # num_pages is num_tokens.

        k_cache = k_cache.view(storage_shape)
        v_cache = v_cache.view(storage_shape)

        for req in batch.reqs:
            q_len = req.extend_len
            current_q = q[start_idx : start_idx + q_len]  # [L_q, H_qo, D]

            logical_len = req.device_len
            physical_indices = self.page_table[req.table_idx, :logical_len].to(torch.long)

            current_k = k_cache[physical_indices]  # [L_kv, H_kv, D]
            current_v = v_cache[physical_indices]  # [L_kv, H_kv, D]

            # GQA expansion
            n_qo = current_q.shape[1]
            n_kv = current_k.shape[1]
            n_rep = n_qo // n_kv

            if n_rep > 1:
                current_k = current_k.repeat_interleave(n_rep, dim=1)
                current_v = current_v.repeat_interleave(n_rep, dim=1)

            # Prepare for SDPA: [Batch, Heads, SeqLen, Dim]
            q_in = current_q.transpose(0, 1).unsqueeze(0)  # [1, H, L_q, D]
            k_in = current_k.transpose(0, 1).unsqueeze(0)  # [1, H, L_kv, D]
            v_in = current_v.transpose(0, 1).unsqueeze(0)  # [1, H, L_kv, D]

            use_causal = False
            if q_len > 1 and q_len == logical_len:
                use_causal = True

            # Handle MPS specific limitation if necessary (MPS supports SDPA now)
            try:
                o = F.scaled_dot_product_attention(q_in, k_in, v_in, is_causal=use_causal)
            except RuntimeError:
                # Fallback for MPS/CPU if SDPA fails (e.g. strict causal mask reqs)
                # Or just basic attention
                scale = 1.0 / (self.config.head_dim**0.5)
                attn = (q_in @ k_in.transpose(-2, -1)) * scale
                if use_causal:
                    mask = torch.ones(q_len, logical_len, device=q.device, dtype=torch.bool)
                    mask = torch.tril(mask)
                    attn = attn.masked_fill(~mask, float("-inf"))
                attn = F.softmax(attn, dim=-1)
                o = attn @ v_in

            outputs.append(o.squeeze(0).transpose(0, 1))  # [L_q, H, D]

            start_idx += q_len

        return torch.cat(outputs, dim=0).view(-1, self.kvcache.num_qo_heads * self.config.head_dim)
