import torch
from rich.progress import track
from torch import nn
from torch.nn import functional as F

n_embd = 32

with open("input.txt", "r") as fr:
    text = fr.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

# 'hello'
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[i] for i in l])
print(decode(encode("hello, I am learning GPT from scratch")))

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 4


def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, size=(batch_size,))

    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    return x, y


x, y = get_batch("train")
for b in range(batch_size):
    for t in range(block_size):
        xb = x[b, : t + 1].tolist()
        yb = y[b, t].tolist()
        print(f"when context is {xb}, the output is: {yb}")
    print("----------")


class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embd_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.emb_table(idx)  # B X T X C,  C => n_embd
        positional_emb = self.positional_embd_table(torch.arange(T))  # TxC
        x = token_emb + positional_emb
        logits = self.lm_head(x)  # BxTx Vocab_size
        
        B, T, C = logits.shape
        loss = None
        if targets is not None:  # targets -> B x T
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = self.loss_fn(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=100):  # BT
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # BTC
            logits = logits[:, -1, :]  # BC
            probs = F.softmax(logits, dim=-1)  # BC
            idx_next = torch.multinomial(probs, num_samples=1)  # B
            idx = torch.cat([idx, idx_next], dim=1)
        print(idx.shape)
        return idx


model = BigramLanguageModel()
logits, loss = model(x, y)
print(decode(model.generate(x)[0].tolist()))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for steps in track(range(10000)):
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps%100 == 0:
        print(f"step: {steps}, loss: {loss}")

print(loss.item())
print(decode(model.generate(torch.tensor([[0]]))[0].tolist()))
