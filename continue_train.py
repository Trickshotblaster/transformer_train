import tiktoken
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import random


enc = tiktoken.get_encoding("gpt2")
enc.decode(enc.encode("Hello world!"))

# #!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# with open('input.txt', 'r', encoding='utf-8') as f: # input.txt
#     text = f.read()
# print(text[:100])

# train_amount = 0.95
# idx = int(train_amount * len(text))
# train_text = text[:idx]
# val_text = text[idx:]

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import random
class DataLoader:
  def __init__(self, text, batch_size, block_size, random_sample=True):
    self.text = torch.tensor(enc.encode(text)).to(device)
    self.batch_size = batch_size
    self.block_size = block_size
    self.current_pos = 0
    self.random_sample = random_sample
  def steps_per_epoch(self):
    return len(self.text) // (self.batch_size * self.block_size)
  def next(self):
    if self.current_pos + self.batch_size * self.block_size + 1 >= len(self.text):
      self.current_pos = 0
    if self.random_sample:
      idx = int((random.random() * len(self.text)) - (self.batch_size * self.block_size + 1) - 1)
      buf = self.text[idx:idx + (self.batch_size * self.block_size + 1)] #[self.current_pos:self.current_pos + self.batch_size * self.block_size + 1]
      if len(buf) == 0:
        return self.next()
    else:
      buf = self.text[self.current_pos:self.current_pos + self.batch_size * self.block_size + 1]
    ins = buf[:-1].view(self.batch_size, self.block_size)
    tgts = buf[1:].view(self.batch_size, self.block_size)
    self.current_pos += self.batch_size * self.block_size + 1
    return ins, tgts



def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class ShardDataLoader:
    def __init__(self, split, batch_size, block_size, start_folder="Data/edu_fineweb10B", random_sample=False):
        self.random_sample = random_sample
        assert split in {'val', 'train'}, 'split must be train or val'
        shards = os.listdir(start_folder)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(start_folder, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
    
        self.split = split
        self.block_size = block_size
        self.batch_size = batch_size
        self.current_shard = 0
        self.current_pos = 0
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.text = load_tokens(self.shards[self.current_shard])
        self.text = self.text.to(device)
        self.current_pos = self.batch_size * self.block_size
    def steps_per_shard(self):
        return len(self.text) // (self.batch_size * self.block_size)
    def next(self):
        if self.random_sample:
           self.current_pos = random.randint(0, len(self.text))
        if self.current_pos + self.batch_size * self.block_size + 1 >= len(self.text):
            self.current_pos = 0
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.text = load_tokens(self.shards[self.current_shard])
            self.text = self.text.to(device)
        buf = self.text[self.current_pos:self.current_pos + self.batch_size * self.block_size + 1]
        ins = buf[:-1].view(self.batch_size, self.block_size)
        tgts = buf[1:].view(self.batch_size, self.block_size)
        self.current_pos += self.batch_size * self.block_size + 1
        return ins, tgts

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    self.d_key = self.d_model // self.n_heads

    self.wq = nn.Linear(d_model, d_model)
    self.wk = nn.Linear(d_model, d_model)
    self.wv = nn.Linear(d_model, d_model)

    self.wo = nn.Linear(d_model, d_model)
  def forward(self, ins, mask=None):
    batch_size, seq_len, d_model = ins.size()
    Q = self.wq(ins).view(batch_size, seq_len, self.n_heads, self.d_key).transpose(1, 2)
    K = self.wk(ins).view(batch_size, seq_len, self.n_heads, self.d_key).transpose(1, 2)
    V = self.wv(ins).view(batch_size, seq_len, self.n_heads, self.d_key).transpose(1, 2)

    #scaled_dot_product = (Q @ K.transpose(2, 3)) / (self.d_model ** 0.5)

    #if mask is not None:
      #scaled_dot_product += mask

    attn_scores = F.scaled_dot_product_attention(Q, K, V, is_causal=True, attn_mask=mask)
    #F.softmax(scaled_dot_product, dim=-1) @ V
    attn_scores = attn_scores.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return self.wo(attn_scores)

class MLP(nn.Module):
  def __init__(self, in_size, hidden_size, out_size):
    super().__init__()
    self.l1 = nn.Linear(in_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, out_size)
    self.gelu = nn.GELU()
  def forward(self, ins):
    acts = self.gelu(self.l1(ins))
    return self.l2(acts)

class DecoderBlock(nn.Module):
  def __init__(self, vocab_size, d_model, n_heads, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.dropout = nn.Dropout(dropout)
    self.MHA = MultiHeadAttention(d_model, n_heads)
    self.MLP = MLP(d_model, 4*d_model, d_model)
    self.layernorm1 = nn.LayerNorm(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)
  def forward(self, ins, mask=None):
    ins = ins + self.MHA(self.layernorm1(ins), mask=mask)
    ins = ins + self.MLP(self.layernorm2(ins))
    return self.dropout(ins)

class GPT(nn.Module):
  def __init__(self, vocab_size, block_size, n_layers=2, n_heads=4, d_model=64, dropout=0.1):
    super().__init__()
    self.vocab_size = vocab_size
    self.block_size = block_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.d_model = d_model
    self.dropout = dropout

    self.token_embedding = nn.Embedding(vocab_size, d_model)
    self.position_embedding = nn.Embedding(block_size, d_model)
    self.decoder_stack = nn.ModuleList([
        DecoderBlock(vocab_size, d_model, n_heads, dropout=dropout) for _ in range(n_layers)
    ])
    self.final_ln = nn.LayerNorm(d_model)
    self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
    #self.token_embedding.weight = self.output_proj.weight 
  def forward(self, ins, targets=None):
    B, T = ins.size()

    x = self.token_embedding(ins.to(device))
    input_indices = torch.arange(T).to(device)
    x += self.position_embedding(input_indices)

    #look_ahead_mask = torch.triu(
        #torch.ones((T, T)), diagonal=1
    #)
    #look_ahead_mask.masked_fill_(look_ahead_mask == 1, float("-inf"))
    #look_ahead_mask = look_ahead_mask.to(device)

    for decoder in self.decoder_stack:
      x = decoder(x) #mask=look_ahead_mask
    x = self.final_ln(x)
    logits = self.output_proj(x)
    loss = None
    if targets is not None:
      targets = targets.to(device)
      loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
    return logits, loss

batch_size = 8
total_batch_size = 16
assert total_batch_size % batch_size == 0, "batch size must be divisible by micro batch size"
grad_accum_steps = total_batch_size // batch_size
block_size = 256
n_layers = 16
n_heads = 12
d_model = 768 * 2


# 64, 32, 8, 8, 128, 3e-4 = best results

my_GPT = GPT(enc.n_vocab, block_size, n_layers, n_heads, d_model, dropout=0.0) #enc.n_vocab
from torchao.float8 import (
    convert_to_float8_training,
    precompute_float8_dynamic_scale_for_fsdp,
)

# optional: filter modules from being eligible for float8 conversion
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "output":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True

# convert all `torch.nn.Linear` modules to `Float8Linear`
convert_to_float8_training(my_GPT, module_filter_fn=module_filter_fn)
my_GPT = my_GPT.to(device)

compile = True
if compile and torch.cuda.is_available():
  my_GPT = torch.compile(my_GPT, mode='reduce-overhead')
  print("Using Compile!!!!!")

my_GPT.load_state_dict(torch.load("latest_model_finetune.pth"))

optim = torch.optim.AdamW(my_GPT.parameters(), lr=3e-4, fused=True)
print("Optimizer created")
data_loader = ShardDataLoader("train", batch_size, block_size, start_folder="Data/instruct-code", random_sample=True)
print("Train data loader created")

val_data_loader = ShardDataLoader("val", batch_size, block_size, start_folder="Data/instruct-code", random_sample=True)
print("Val data loader created")
val_interval = 1000

log_interval = 50
max_steps = 10000
val_loss_steps = 4
print("Steps per epoch:", data_loader.steps_per_shard())
print(f"GPT Parameters: {sum(p.numel() for p in my_GPT.parameters()) / 1e6} million")

torch.set_float32_matmul_precision("high")

import math
max_lr = 2e-4
min_lr = max_lr * 0.1
warmup_steps = 715

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


import time



best_val_loss = float("inf")
my_GPT.train()
for step in range(max_steps + 1):
  step_start = time.time()
  optim.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = data_loader.next()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      logits, loss = my_GPT(x, y)
  
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    loss.backward()
    
  norm = torch.nn.utils.clip_grad_norm_(my_GPT.parameters(), 1.0)
  lr = get_lr(step)
  for param_group in optim.param_groups:
      param_group['lr'] = lr
  optim.step()


  if step % log_interval == 0:
    torch.cuda.synchronize()
    step_time = time.time() - step_start
    print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {step_time*1000:.2f}ms | tok/s: {(batch_size * block_size * grad_accum_steps) / step_time}")
  if step % val_interval == 0:
    with torch.no_grad():
      my_GPT.eval()
      for run in range(5):
        prompt = "The capital of France is"
        input_tokens = enc.encode(prompt)
        output_tokens = enc.encode(prompt)
        top_k = 50
        for x in range(200):
          if len(input_tokens) > block_size:
            input_tokens = input_tokens[1:]
          context_tensor = torch.tensor(input_tokens).view(1, -1).to(device)
          with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = my_GPT(context_tensor)
          logits = logits[:, -1, :]
          if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")
          probs = F.softmax(logits, dim=-1)
          result = torch.multinomial(probs, num_samples=1).item()
          input_tokens.append(result)
          output_tokens.append(result)
        try:
          print(enc.decode(output_tokens))
        except:
          print("oops we predicted a fake token")
      val_loss = 0
      for val_step in range(val_loss_steps):
        val_x, val_y = val_data_loader.next()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
          logits, loss = my_GPT(val_x, val_y)
        val_loss += loss
      val_loss /= val_loss_steps

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(my_GPT.state_dict(), 'best_model_finetune_cont.pth')
      torch.save(my_GPT.state_dict(), 'latest_model_finetune_cont.pth')
      print(f"Val loss for step {step}: {val_loss}")
      my_GPT.train()

try:
  with torch.no_grad():
    my_GPT.eval()
    for run in range(5):
      prompt = "The capital of France is"
      input_tokens = enc.encode(prompt)
      output_tokens = enc.encode(prompt)
      top_k = 50
      for x in range(200):
        if len(input_tokens) > block_size:
          input_tokens = input_tokens[1:]
        context_tensor = torch.tensor(input_tokens).view(1, -1).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
          logits, loss = my_GPT(context_tensor)
        logits = logits[:, -1, :]
        if top_k > 0:
              # Remove all tokens with a probability less than the last token of the top-k
              indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
              logits[indices_to_remove] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        result = torch.multinomial(probs, num_samples=1).item()
        input_tokens.append(result)
        output_tokens.append(result)
      print(enc.decode(output_tokens))
except:
   print("oops")