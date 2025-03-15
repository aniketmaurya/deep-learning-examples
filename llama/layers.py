import torch
import torch.nn.functional as F
from torch.nn import RMSNorm
from dataclasses import dataclass
from typing import Optional
from torch import nn


@dataclass
class ModelArgs:
    device: str = "cpu"
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 1.3
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    max_batch_size: int = 2
    max_seq_len: int = 2048
    use_scaled_rope = False


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (
        x.shape[1], x.shape[-1]), f"shape mismatch {freqs_cis.shape}, {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: FOLLOW UP
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        # Use hidden_dim directly as intermediate_size (8192 for Llama-3.2-1B)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.head_dim = params.dim // params.n_heads
        self.n_reps = params.n_heads // params.n_kv_heads
        self.n_kv_heads = params.n_kv_heads
        self.n_heads = params.n_heads
        self.dim = params.dim
        self.device = params.device
        self.max_seq_len = params.max_seq_len
        self.max_batch_size = params.max_batch_size

        self.q_proj = nn.Linear(params.dim, params.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(params.dim, params.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(params.dim, params.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(params.n_heads * self.head_dim, params.dim, bias=False)  # Fixed output dim

        self.k_cache = None
        self.v_cache = None

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seq_len, _ = x.shape

        # Initialize or reset cache if needed
        if self.k_cache is None or self.k_cache.shape[0] != bsz or self.k_cache.device != x.device:
            self.k_cache = torch.zeros(bsz, self.max_seq_len, self.n_kv_heads, self.head_dim, device=x.device)
            self.v_cache = torch.zeros(bsz, self.max_seq_len, self.n_kv_heads, self.head_dim, device=x.device)

        # Projections and reshaping
        xq = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Update cache
        self.k_cache[:bsz, start_pos:start_pos + seq_len] = xk
        self.v_cache[:bsz, start_pos:start_pos + seq_len] = xv

        # Retrieve full keys and values
        xk = self.k_cache[:bsz, :start_pos + seq_len]
        xv = self.v_cache[:bsz, :start_pos + seq_len]

        # Repeat KV for GQA
        xk = torch.repeat_interleave(xk, dim=2, repeats=self.n_reps)
        xv = torch.repeat_interleave(xv, dim=2, repeats=self.n_reps)

        # Attention computation
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)  # (bsz, n_heads, seq_len + start_pos, head_dim)
        xv = xv.transpose(1, 2)  # (bsz, n_heads, seq_len + start_pos, head_dim)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / (
                self.head_dim ** 0.5)  # (bsz, n_heads, seq_len, seq_len + start_pos)
        if mask is not None:
            scores = scores + mask  # Broadcasting needs mask shape (seq_len, start_pos + seq_len)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bsz, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)  # (bsz, seq_len, dim)
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.self_attn = Attention(params)
        self.mlp = MLP(params.dim, 4 * params.dim, params.multiple_of, params.ffn_dim_multiplier)
        self.input_layernorm = nn.RMSNorm(params.dim, eps=params.norm_eps)
        self.post_attention_layernorm = RMSNorm(params.dim, eps=params.norm_eps)

    def forward(self, x, start_pos, freq_cis, mask):
        h = x + self.self_attn(self.input_layernorm(x), start_pos, freq_cis, mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.embed_tokens = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList([TransformerBlock(params) for _ in range(params.n_layers)])
        self.norm = nn.RMSNorm(params.dim, eps=params.norm_eps)
        self.lm_head = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.lm_head(h).float()
        return output
