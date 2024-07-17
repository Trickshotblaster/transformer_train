import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, Repository
from getpass import getpass
import os
import shutil
import tempfile

# Original model classes (simplified for brevity)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = self.d_model // self.n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, ins, mask=None):
        # Simplified forward pass
        return self.wo(ins)

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = nn.Linear(in_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)
        self.gelu = nn.GELU()

    def forward(self, ins):
        return self.l2(self.gelu(self.l1(ins)))

class DecoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, n_heads)
        self.MLP = MLP(d_model, 4*d_model, d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ins, mask=None):
        x = ins + self.MHA(self.layernorm1(ins), mask=mask)
        x = x + self.MLP(self.layernorm2(x))
        return self.dropout(x)

class OriginalGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layers=2, n_heads=4, d_model=64, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.decoder_stack = nn.ModuleList([
            DecoderBlock(vocab_size, d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ins, targets=None):
        # Simplified forward pass
        return None, None

# Hugging Face compatible model classes
class GPTConfig(PretrainedConfig):
    model_type = "gpt"
    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_layers=12,
        n_heads=12,
        d_model=768,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout

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

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.size()
        x = self.token_embedding(input_ids)
        input_indices = torch.arange(T, device=input_ids.device)
        x += self.position_embedding(input_indices)
        for decoder in self.decoder_stack:
            x = decoder(x)
        x = self.final_ln(x)
        logits = self.output_proj(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return {'logits': logits, 'loss': loss}

    def generate(self, input_ids, max_length, num_return_sequences=1, top_k=50):
        B, T = input_ids.size()
        generated = input_ids
        for _ in range(max_length - T):
            if generated.size(1) > self.config.block_size:
                generated = generated[:, -self.config.block_size:]
            outputs = self(generated)
            logits = outputs['logits'][:, -1, :]
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

def convert_model(original_model_path, new_model_path):
    print("Converting model...")
    original_model = OriginalGPT(vocab_size=50257, block_size=1024, n_layers=12, n_heads=12, d_model=768)
    original_model.load_state_dict(torch.load(original_model_path), strict=False)

    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layers=12,
        n_heads=12,
        d_model=768,
        dropout=0.1
    )

    new_model = GPT(config)

    new_model.token_embedding.weight.data = original_model.token_embedding.weight.data
    new_model.position_embedding.weight.data = original_model.position_embedding.weight.data

    for i in range(12):
        new_model.decoder_stack[i].MHA.wq.weight.data = original_model.decoder_stack[i].MHA.wq.weight.data
        new_model.decoder_stack[i].MHA.wk.weight.data = original_model.decoder_stack[i].MHA.wk.weight.data
        new_model.decoder_stack[i].MHA.wv.weight.data = original_model.decoder_stack[i].MHA.wv.weight.data
        new_model.decoder_stack[i].MHA.wo.weight.data = original_model.decoder_stack[i].MHA.wo.weight.data
        
        new_model.decoder_stack[i].MLP.l1.weight.data = original_model.decoder_stack[i].MLP.l1.weight.data
        new_model.decoder_stack[i].MLP.l2.weight.data = original_model.decoder_stack[i].MLP.l2.weight.data
        
        new_model.decoder_stack[i].layernorm1.weight.data = original_model.decoder_stack[i].layernorm1.weight.data
        new_model.decoder_stack[i].layernorm2.weight.data = original_model.decoder_stack[i].layernorm2.weight.data

    new_model.final_ln.weight.data = original_model.final_ln.weight.data
    new_model.output_proj.weight.data = original_model.output_proj.weight.data

    new_model.save_pretrained(new_model_path)
    print(f"Model converted and saved to {new_model_path}")

def create_tokenizer(save_path):
    print("Creating tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")

def push_to_hub(local_dir, repo_name):
    print("Pushing to Hugging Face Hub...")
    api = HfApi()
    #token = getpass("Enter your Hugging Face API token: ")
    #api.set_token(token)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory: {tmp_dir}")
        
        # Copy files to temporary directory
        for item in os.listdir(local_dir):
            s = os.path.join(local_dir, item)
            d = os.path.join(tmp_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        
        # Initialize git repo in the temporary directory
        repo = Repository(
            local_dir=tmp_dir,
            clone_from=None,
            #use_auth_token=token,
        )
        
        # Create the remote repository
        repo_url = api.create_repo(
            repo_id=repo_name,
            private=False,
            exist_ok=True,
        )
        
        # Add the remote to the local repository
        repo.git_remote_add_url(repo_url)
        
        # Push to hub
        repo.push_to_hub(commit_message="Initial commit", use_temp_dir=True)
        
    print(f"Model pushed to {repo_url}")




def test_model(model_name):
    print(f"Testing model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "ROMEO:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, top_k=50)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:")
    print(generated_text)

def main():
    original_model_path = 'latest_model_finetune.pth'
    new_model_path = 'huggingface_gpt_model'
    repo_name = "Trickshotblaster/mike-40k-hf"  # Change this to your desired repository name

    # Convert the model
    convert_model(original_model_path, new_model_path)

    # Create and save the tokenizer
    create_tokenizer(new_model_path)

    # Create README.md
    readme_content = """
    # GPT Model

    This is a GPT model fine-tuned on custom data.

    ## Usage

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "your-username/your-model-name"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "ROMEO:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, top_k=50)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    ```

    ## Model Details

    - Architecture: GPT
    - Vocab Size: 50257
    - Hidden Size: 768
    - Number of Layers: 12
    - Number of Attention Heads: 12
    """

    with open(os.path.join(new_model_path, 'README.md'), 'w') as f:
        f.write(readme_content)

    # Push to Hugging Face Hub
    push_to_hub(new_model_path, repo_name)

    # Test the model
    test_model(repo_name)

if __name__ == "__main__":
    main()