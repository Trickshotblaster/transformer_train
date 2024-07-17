import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModelForCausalLM
from huggingface_hub import PyTorchModelHubMixin

import tiktoken
enc = tiktoken.get_encoding("gpt2")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GPTConfig(PretrainedConfig):
    model_type = "custom_gpt"
    def __init__(
        self,
        vocab_size=50257,
        block_size=512,
        n_layers=12,
        n_heads=12,
        d_model=768,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout


class MultiHeadAttention(nn.Module, PyTorchModelHubMixin):
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

class MLP(nn.Module, PyTorchModelHubMixin):
  def __init__(self, in_size, hidden_size, out_size):
    super().__init__()
    self.l1 = nn.Linear(in_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, out_size)
    self.gelu = nn.GELU()
  def forward(self, ins):
    acts = self.gelu(self.l1(ins))
    return self.l2(acts)

class DecoderBlock(nn.Module, PyTorchModelHubMixin):
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

class GPT(PreTrainedModel):
    config_class = GPTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.decoder_stack = nn.ModuleList([
            DecoderBlock(config.vocab_size, config.d_model, config.n_heads, dropout=config.dropout) 
            for _ in range(config.n_layers)
        ])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def forward(self, ins, targets=None):
        B, T = ins.size()

        x = self.token_embedding(ins)
        input_indices = torch.arange(T, device=ins.device)
        x += self.position_embedding(input_indices)

        for decoder in self.decoder_stack:
            x = decoder(x)
        x = self.final_ln(x)
        logits = self.output_proj(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        
        return {"logits": logits, "loss": loss}

    def generate(self, input_ids, max_length, num_return_sequences=1, **kwargs):
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self(input_ids)
            next_token_logits = outputs["logits"][:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

    def load_state_dict(self, state_dict, strict=True):
        # Custom method to load state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('token_embedding.'):
                new_k = k.replace('token_embedding.', 'token_embedding.')
            elif k.startswith('position_embedding.'):
                new_k = k.replace('position_embedding.', 'position_embedding.')
            elif k.startswith('decoder_stack.'):
                new_k = k.replace('decoder_stack.', 'decoder_stack.')
            elif k == 'final_ln.weight':
                new_k = 'final_ln.weight'
            elif k == 'final_ln.bias':
                new_k = 'final_ln.bias'
            elif k == 'output_proj.weight':
                new_k = 'output_proj.weight'
            else:
                new_k = k
            new_state_dict[new_k] = v
        return super().load_state_dict(new_state_dict, strict=strict)

# Register the custom model
AutoConfig.register("custom_gpt", GPTConfig)
AutoModelForCausalLM.register(GPTConfig, GPT)

print("GPT class initialized")

# Create a configuration
config = GPTConfig(
    vocab_size=enc.n_vocab,
    block_size=512,
    n_layers=12,
    n_heads=12,
    d_model=768,
    dropout=0.0
)

# Create the model
my_GPT = GPT(config)
print("empty model created")

# Load the state dict
state_dict = torch.load('latest_model_finetune.pth')
my_GPT.load_state_dict(state_dict, strict=False)
my_GPT.eval()
print("GPT loaded")

# Save the model and tokenizer
from transformers import AutoTokenizer

# Initialize the tokenizer (GPT-2 tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Save the model and tokenizer
my_GPT.save_pretrained("./my_gpt_model")
tokenizer.save_pretrained("./my_gpt_model")

# Push to the hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./my_gpt_model",
    repo_id="Trickshotblaster/mike",
    repo_type="model",
)

print("pushed to hub")

# To use the model:
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Trickshotblaster/mike")
tokenizer = AutoTokenizer.from_pretrained("Trickshotblaster/mike")

# Example usage
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

print("done")