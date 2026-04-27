import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from pathlib import Path

import tiktoken
import string
import time


class DataManager:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.text = self.data_path.read_text(encoding="utf-8")
        self.device = self._get_device()
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.vocab_size = self.tokenizer.n_vocab
        self.train_data, self.test_data = self._split_data()

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.xpu.is_available():
            return torch.device("xpu")
        else:
            return torch.device("cpu")

    def _tokenize_vocab(
        self,
    ) -> tuple[int, dict[str, int], dict[int, str]]:
        chars = sorted(list(set(string.printable + self.text)))
        vocab_size = len(chars)
        char_map = {c: i for i, c in enumerate(chars)}
        int_map = {i: c for i, c in enumerate(chars)}

        return vocab_size, char_map, int_map

    def _split_data(self) -> tuple[torch.Tensor, torch.Tensor]:

        encoded_text = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(len(encoded_text) * 0.9)

        train_data = encoded_text[:n]
        test_data = encoded_text[n:]

        return train_data, test_data

    def encode(self, x: str) -> list[int]:
        return self.tokenizer.encode(x)

    def decode(self, x: list[int]) -> str:
        if isinstance (x, torch.Tensor):
            x = x.tolist()
        return self.tokenizer.decode(x)

    def get_batch(
        self, mode: str, batch_size=4, block_size=8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "train":
            data = self.train_data
        elif mode == "test":
            data = self.test_data

        ix = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        return x.to(self.device), y.to(self.device)


class Head(nn.Module):
    def __init__(self, head_size: int, embed_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(0.2)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, head_size: int, embed_size: int, block_size: int
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, embed_size, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, block_size: int):
        super().__init__()
        head_size = embed_size // num_heads

        self.sa = MultiHeadAttention(num_heads, head_size, embed_size, block_size)
        self.ffwd = FeedForward(embed_size)

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explain the x + more
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class NanoGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)

        self.blocks = nn.Sequential(
            *[Block(embed_size, num_heads, block_size) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)

        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)

            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, block_size: int, temp: float = 1.0
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]

            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temp

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


def train(
    model: NanoGPT,
    dm: DataManager,
    optimizer: optim.Optimizer,
    batch_size: int,
    block_size: int,
    max_steps: int,
    eval_interval: int,
    eval_iters: int,
) -> None:
    print("Starting Training")
    start_time = time.perf_counter()
    threshold = float('inf')
    save_weights: dict[str, torch.Tensor] = model.state_dict()
    for step in range(max_steps):
        if step == 0 or (step + 1) % eval_interval == 0:
            losses = estimate_loss(model, dm, eval_iters, batch_size, block_size)
            print(
                f"Step:{step + 1} | Training Loss: {losses['train']:.4f} | Testing Loss: {losses['test']:.4f}"
            )

        x, y = dm.get_batch("train", batch_size, block_size)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < threshold:
            threshold = loss.item()
            save_weights = model.state_dict()

    print(f"Training Completed. Elapsed Time: {time.perf_counter() - start_time}")
    torch.save(save_weights, 'nanoGPT.pth')
    print(f"Model saved with loss of {threshold}")

@torch.no_grad()
def estimate_loss(
    model: NanoGPT, dm: DataManager, eval_iters: int, batch_size: int, block_size: int
) -> dict[str, float]:
    out = {}
    model.eval()
    for t in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dm.get_batch(t, batch_size, block_size)
            logits, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[t] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    path = r"LSTM\training_data\tiny_shakespear.txt"
    dm = DataManager(path)

    batch_size = 64
    block_size = 64
    max_steps = 500
    eval_interval = 500
    eval_iters = 100

    model = NanoGPT(
        dm.vocab_size, embed_size=256, block_size=block_size, num_heads=4, num_layers=4
    ).to(dm.device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(
        model, dm, optimizer, batch_size, block_size, max_steps, eval_interval, eval_iters
    )

    print("\nGenerating New Text")
    context = torch.tensor([dm.encode("\n")], dtype=torch.long, device=dm.device)

    generated_indices = model.generate(context, 500, block_size)

    print(dm.decode(generated_indices[0]))
