import torch
from rich.progress import track
from torch import nn
from torch.nn import functional as F

n_embd = 32
eval_iters = 100
eval_interval = 1000
max_iter = 10000
block_size = 8
batch_size = 32
device = "cpu"

with open("input.txt", "r") as fr:
    text = fr.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

# 'hello'
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, size=(batch_size,))

    x = torch.stack([data[i : i + block_size] for i in idx]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters, device=device)
        for i in range(eval_iters):
            x, y = get_batch(split=split)
            logits, loss = model(x, y)
            losses[i] = loss
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size, device=device)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B x T x head_size
        q = self.query(x)  # B x T x head_size
        
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


# class AvgHead(nn.Module):
#     def __init__(self, head_size=None):
#         super().__init__()
#         self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
#     def forward(self, x):
#         B, T, C = x.shape
#         wei = torch.tril(torch.ones(T, T))
#         wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))  # (B,T,T)
#         wei = F.softmax(wei, dim=-1)  # (B, T, T)
#         out = wei @ x  # (B, T, C)
#         return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, idx):
        return torch.cat([head(idx) for head in self.heads], dim=-1)

class Block(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.sa_head = MultiHeadAttention(head_size, n_embd//head_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):  # (B,T,C)
        x = self.sa_head(x)
        x = self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embd_table = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(
            Block(n_embd=n_embd, head_size=4),
            Block(n_embd=n_embd, head_size=4),
            Block(n_embd=n_embd, head_size=4),
            Block(n_embd=n_embd, head_size=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.emb_table(idx)  # B X T X C,  C => n_embd
        positional_emb = self.positional_embd_table(torch.arange(T, device=device))  # TxC
        x = token_emb + positional_emb
        x = self.block(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        B, T, C = logits.shape
        loss = None
        if targets is not None:  # targets -> B x T
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = self.loss_fn(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=500):  # BT
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # BTC
            logits = logits[:, -1, :]  # BC
            probs = F.softmax(logits, dim=-1)  # BC
            idx_next = torch.multinomial(probs, num_samples=1)  # B
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


model = BigramLanguageModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


for iter in track(range(max_iter), description="Training..."):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context)[0].cpu().tolist()))
