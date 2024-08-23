import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken
enc = tiktoken.get_encoding("gpt2")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


block_size = 256
n_layers = 16
n_heads = 12
d_model = 768 * 2

torch.set_float32_matmul_precision('high')

my_GPT = GPT(enc.n_vocab, block_size, n_layers, n_heads, d_model, dropout=0.1) #enc.n_vocab
# from torchao.float8 import (
#     convert_to_float8_training,
#     precompute_float8_dynamic_scale_for_fsdp,
# )

# # optional: filter modules from being eligible for float8 conversion
# def module_filter_fn(mod: torch.nn.Module, fqn: str):
#     # don't convert the output module
#     if fqn == "output":
#         return False
#     # don't convert linear modules with weight dimensions not divisible by 16
#     if isinstance(mod, torch.nn.Linear):
#         if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
#             return False
#     return True

# # convert all `torch.nn.Linear` modules to `Float8Linear`
# convert_to_float8_training(my_GPT, module_filter_fn=module_filter_fn)
my_GPT = my_GPT.to(device)
my_GPT = torch.compile(my_GPT)
my_GPT.load_state_dict(torch.load('latest_model_finetune_cont.pth'))
my_GPT.eval()

eot = enc._special_tokens['<|endoftext|>']

while True:
  pass
  prompt = "USER: " + input("Prompt:") + "\nASSISTANT: "
  if prompt.find("quit") != -1:
    break
  input_tokens = enc.encode(prompt)
  output_tokens = enc.encode(prompt)
  top_k = 50
  top_p = 0
  for x in range(block_size):
    if len(input_tokens) > block_size:
      input_tokens = input_tokens[1:]
    context_tensor = torch.tensor(input_tokens).view(1, -1).to(device)

    logits, loss = my_GPT(context_tensor)
    logits = logits[:, -1, :] 
    if top_k > 0:
          # Remove all tokens with a probability less than the last token of the top-k
          indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
          logits[indices_to_remove] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    result = torch.multinomial(probs, num_samples=1).item()
    if result == eot:
      break
    input_tokens.append(result)
    output_tokens.append(result)
    
  print(enc.decode(output_tokens))