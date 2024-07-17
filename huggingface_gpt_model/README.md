
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
    