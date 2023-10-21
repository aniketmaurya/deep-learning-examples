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

    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
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
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
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


class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embd_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.emb_table(idx)  # B X T X C,  C => n_embd
        positional_emb = self.positional_embd_table(torch.arange(T))  # TxC
        x = token_emb + positional_emb
        x = self.sa_head(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        B, T, C = logits.shape
        loss = None
        if targets is not None:  # targets -> B x T
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = self.loss_fn(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=200):  # BT
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # BTC
            logits = logits[:, -1, :]  # BC
            probs = F.softmax(logits, dim=-1)  # BC
            idx_next = torch.multinomial(probs, num_samples=1)  # B
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


model = BigramLanguageModel()
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
print(decode(model.generate(context)[0].tolist()))
