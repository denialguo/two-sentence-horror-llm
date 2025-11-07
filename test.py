import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "microsoft/phi-3-mini-128k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


adapter_path = "/kaggle/input/horror-story-generator-v1/horror-model"

# This merges your adapter with the base model
tuned_model = PeftModel.from_pretrained(base_model, adapter_path)

print("✅ Model loaded successfully!")

# put in a prompt
prompt = "I always wave at the little girl in the window across the street."

inputs = tokenizer(
    f"<|user|>\n{prompt}<|end|>\n<|assistant|>", 
    return_tensors="pt", 
    return_attention_mask=False
).to("cuda")

outputs = tuned_model.generate(
    **inputs,
    max_new_tokens=60,
    temperature=0.7,
    do_sample=True,
    use_cache=False,  # <-- THIS IS THE FIX
)


full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

assistant_response = full_response.split("<|assistant|>")[-1]
print("\n---")
print(f"Prompt: {prompt}")
print(f"Model: {assistant_response.strip()}")
