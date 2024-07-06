import torch
from torch import tensor

with open("input.txt") as file:
    text = file.read()

# unique characters that occur in the text
chars = sorted(set(text))
vocab_size = len(chars)

# tokenize: convert a string to a sequence of integers according to some vocabulary
# Google uses sentencepiece (sub-word units)
# OpenAI uses tiktoken (byte pair encoding)

# encoder and decoder
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# text represented as torch.Tensor
t = tensor(encode(text), dtype=torch.long)
print(t.shape)
print(t.dtype)

# split up the data into train and validation sets
