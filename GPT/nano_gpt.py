import torch
from torch import nn
from pathlib import Path

import kagglehub
import string


class DataManager:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.text = self.data_path.read_text(encoding='utf-8')
        self.vocab_size, self.char_map, self.int_map = self._tokenize_vocab()
        self.train_data, self.test_data = self._split_data()

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
        return [self.char_map[c] for c in x]

    def decode(self, x: torch.Tensor) -> str:
        return "".join([self.int_map[i.item()] for i in x])

    def get_batch(self, mode: str, batch_size=4, block_size=8) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "train":
            data = self.train_data
        elif mode == "eval":
            data = self.test_data

        ix = torch.randint(len(data) - block_size, (batch_size,))

        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        return x, y


class Head(nn.Module):

    def __init__(self, head_size: int, embed_size:int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v

        return out


class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, block_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)

    def forward(self, idx: torch.Tensor)->torch.Tensor:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        x = tok_emb + pos_emb
        return x


if __name__ == "__main__":
    path = r"LSTM\training_data\tiny_shakespear.txt"
    dm = DataManager(path)
    model = NanoGPT(dm.vocab_size, embed_size=256, block_size=8)
    x, y = dm.get_batch('train')
    print(model(x).shape)


