import torch
import torch.functional as F
import torch.nn as nn

# Input shape: (batch_size, seq_len, hidden_dim) or B x T x C
# Output shape: (batch_size, seq_len, hidden_dim) or B x T x C

device = "cuda" if torch.cuda.is_available() else "cpu"

class Config:
    context_length=1024
    embedding_dim=512
    num_heads=4


class AddNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_norm = nn.LayerNorm()

    def forward(self, x1, x2):
        return self.layer_norm(x1+x2)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_size = Config.embedding_dim // Config.num_heads
        T, C = Config.context_length, Config.embedding_dim
        self.attn = nn.Linear(C, T*C*3, bias=False, device=device)
        self.softmax = nn.Softmax()
        self.register_buffer(
            "tril", torch.tril(torch.ones(Config.context_length, Config.context_length, device=device))
        )

    
    def forward(self, x):
        B, T, C = x.size()
        kqv = self.attn(x).view(B, T, C)
        k, q, v = torch.split(kqv, 1, dim=-1)

        # k: BxTxC
        # q: BxTxC
        # v: BxTxC

        # QK dim:     BxTxT
        # QK * T dim: BxTxC

        wei = ((q @ k.transpose(-2, -1))* Config.embedding_dim**-0.5)  # BxTxT
        wei = wei.masked_fill(self.trill[:T, :T]==0, float("-inf"))
        wei = F.softmax(wei)
        wei = self.dropout(wei)
        out = wei * v
        return out


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mha = MultiHeadAttention()
        self.add_norm = AddNorm()
        self.ffw = nn.Linear(Config.context_length, Config.embedding_dim)

    def forward(self, x):
        # x shape => B x T x C
        x1 = self.mha(x)
        x = self.add_norm(x1, x)
        return self.add_norm(self.ffw(x), x)


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.context_length, config.embedding_dim)
        self.pos_encoding = nn.Embedding(config.context_length, config.embedding_dim)
        self.block = [Block() for _ in range(12)]
        self.ff = nn.Linear(config.context_length, config.embedding_dim)
        self.softmax = nn.Softmax()

    
    def forward(self, x):
        # shape of x => BxT
        # B => batch size
        # T => sequence length
        B, T = x.size()
        x = self.embedding(x)  # B x T x C
        pos = self.pos_encoding(torch.arange(T, self.config.context_length))  # B x T
        x = x + pos  # B x T x C

        x = self.block(x)
        x = self.ff(x)
        return self.softmax(x)
