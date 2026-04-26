import torch
from torch import nn

import kagglehub
import string

def tokenize()->tuple[int, dict, dict]:
    chars = sorted(list(set(string.printable)))
    vocab_size = len(chars)
    print(chars)
    char_map = {c: i for i, c in enumerate(chars)}
    int_map = {i: c for i, c in enumerate(chars)}

    return vocab_size, char_map, int_map

def encode(x: str, char_map: dict)->list[int,]:
    return [char_map[c] for c in x]

def decode(x: list, int_map: dict)->str:
    return ''.join([int_map[i] for i in x])

v, c, i = tokenize()
print(encode("Hello World", c))
