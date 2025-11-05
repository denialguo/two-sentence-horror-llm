!pip install --upgrade "transformers[torch]" "datasets" "accelerate" "bitsandbytes" "peft" "trl"


import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import PeftModel, LoraConfig
from trl import SFTTrainer

print("Loading dataset from Hub...")
dataset = load_dataset("yoonholee/reddit_TwoSentenceHorror_3483", split="train")
print(f"Original dataset size: {len(dataset)}")

print("Filtering dataset based on upvote_ratio > 0.85...")
def is_good_story(example):
    is_highly_upvoted = example['upvote_ratio'] > 0.85
    is_long_enough = len(example['text']) > 40
    return is_highly_upvoted and is_long_enough

clean_dataset = dataset.filter(is_good_story)
print(f"Cleaned dataset size: {len(clean_dataset)}")

model_id = "microsoft/phi-3-mini-128k-instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj"],
)

training_args = TrainingArguments(
    output_dir="./horror-model-final",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=25,
    report_to="none",
)

def formatting_func(example):
    """Format dataset examples into text"""
    return example["text"]

trainer = SFTTrainer(
    model=model,
    train_dataset=clean_dataset,
    args=training_args,
    peft_config=peft_config,
    formatting_func=formatting_func,
)

print(f"Training on {len(clean_dataset)} stories...")
trainer.train()
trainer.save_model("./horror-model-final")
print("✅ Model adapter saved to ./horror-model-final")

print("\n--- Loading fine-tuned model for testing ---")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
tuned_model = PeftModel.from_pretrained(base_model, "./horror-model-final")
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=tuned_model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Generating new story...")
prompt = "I always wave at the little girl in the window across the street."
result = pipe(f"{prompt}<|endoftext|>", max_new_tokens=60, do_sample=True, temperature=0.9)

print("\n--- PROMPT ---")
print(prompt)
print("\n--- GENERATED STORY ---")
print(result[0]["generated_text"])
