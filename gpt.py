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
data = tensor(encode(text), dtype=torch.long)

# split up the data into train and validation sets
# 90% of the data will be the training data
# remaining 10% of the data will be validation data
n = int(.9 * len(data)) # 
train_data, val_data = data[:n], data[n:] 

batch_size = 4 # number of independent sequences to be processed in parallel
block_size = 8 # maximum context length to make predictions

# We train them chunks at a time

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")