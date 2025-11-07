# ============================================================================
# 🎃 TWO-SENTENCE HORROR STORY GENERATOR - KAGGLE VERSION 
# Fine-tuning Phi-3 on Reddit r/TwoSentenceHorror stories
# ============================================================================
# to set up kaggle:
# 1. Create new notebook
# 2. Settings → Accelerator → GPU T4 x2
# 3. Settings → Persistence → Files only
# 4. Run this entire notebook
# ============================================================================

# CRITICAL: Set this BEFORE any other imports!
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("📦 Installing required libraries...")
!pip install -q --upgrade transformers datasets accelerate bitsandbytes peft trl

import torch
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig
from trl import SFTTrainer

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Count: {torch.cuda.device_count()}")

# 3. LOAD AND FILTER DATASET
print("\n" + "="*70)
print("📊 LOADING DATASET")
print("="*70)

dataset = load_dataset("yoonholee/reddit_TwoSentenceHorror_3483", split="train")
print(f"Original dataset size: {len(dataset)}")

def is_good_story(example):
    text = example['text']
    is_highly_upvoted = example['upvote_ratio'] > 0.85
    is_long_enough = 40 < len(text) < 500
    has_no_urls = 'http' not in text.lower()
    has_no_meta = '[removed]' not in text.lower() and '[deleted]' not in text.lower()
    return all([is_highly_upvoted, is_long_enough, has_no_urls, has_no_meta])

clean_dataset = dataset.filter(is_good_story)
print(f"✅ Cleaned dataset size: {len(clean_dataset)}")
print(f"✅ Filtered out: {len(dataset) - len(clean_dataset)} low-quality stories")

print("\n📖 Sample story from dataset:")
print(clean_dataset[0]['text'][:200] + "...")

print("\n" + "="*70)
print("🤖 LOADING BASE MODEL")
print("="*70)

model_id = "microsoft/phi-3-mini-128k-instruct"

# 4-bit quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Loading {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0",  # Explicitly use only first GPU
    trust_remote_code=True,
    max_memory={0: "14GB"},  # Limit memory on GPU 0
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded successfully")
print(f"✅ Model size: ~{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")


print("\n" + "="*70)
print("⚙️  CONFIGURING LORA")
print("="*70)

peft_config = LoraConfig(
    r=4,                      # LoRA rank
    lora_alpha=8,             # LoRA alpha (scaling)
    lora_dropout=0.1,         # Dropout probability
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj"],  # Phi-3 attention modules
)

print("✅ LoRA config created")
print(f"   - Rank: {peft_config.r}")
print(f"   - Alpha: {peft_config.lora_alpha}")
print(f"   - Target modules: {peft_config.target_modules}")

print("\n" + "="*70)
print("🎯 TRAINING CONFIGURATION")
print("="*70)

# Kaggle output directory - files here will be saved
output_dir = "/kaggle/working/horror-model"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,      
    learning_rate=2e-4,
    fp16=True,                          
    optim="paged_adamw_8bit",
    logging_steps=25,
    save_steps=400,                   
    save_total_limit=2,                 
    report_to="none",
    warmup_steps=100,
    weight_decay=0.01,
    dataloader_pin_memory=False,        
    ddp_find_unused_parameters=False,   
)

print("✅ Training arguments set")
print(f"   - Epochs: {training_args.num_train_epochs}")
print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   - Learning rate: {training_args.learning_rate}")
print(f"   - Output: {output_dir}")

def formatting_func(example):
    return example["text"]

print("\n" + "="*70)
print("🚀 INITIALIZING TRAINER")
print("="*70)

trainer = SFTTrainer(
    model=model,
    train_dataset=clean_dataset,
    args=training_args,
    peft_config=peft_config,
    formatting_func=formatting_func,
)

print("✅ Trainer initialized")

print("\n" + "="*70)
print(f"🔥 STARTING TRAINING ON {len(clean_dataset)} STORIES")
print("="*70)
print("⏰ This will take approximately 15-20 minutes...")
print()

start_time = time.time()
trainer.train()
training_time = time.time() - start_time

print("\n" + "="*70)
print(f"✅ TRAINING COMPLETE! Time: {training_time/60:.1f} minutes")
print("="*70)

print("\n" + "="*70)
print("💾 SAVING MODEL")
print("="*70)

os.makedirs(output_dir, exist_ok=True)


print("Extracting trained adapter...")

try:
    # Method 1: Try to save directly from trainer
    trainer.model.save_pretrained(output_dir)
    print("✅ Saved via trainer.model")
except Exception as e:
    print(f"⚠️ Direct save failed: {e}")
    print("Trying alternative method...")
    
    try:
        if hasattr(trainer.model, 'base_model'):
            trainer.model.base_model.save_pretrained(output_dir)
            print("✅ Saved via base_model")
        else:
            # Method 3: Manual state dict save
            torch.save(trainer.model.state_dict(), f"{output_dir}/adapter_model.bin")
            print("✅ Saved state dict manually")
    except Exception as e2:
        print(f"❌ All save methods failed: {e2}")

tokenizer.save_pretrained(output_dir)
print("✅ Tokenizer saved")

time.sleep(2)

possible_files = [
    f"{output_dir}/adapter_model.safetensors",
    f"{output_dir}/model.safetensors",
    f"{output_dir}/adapter_model.bin",
    f"{output_dir}/pytorch_model.bin",
]

saved_file = None
for filepath in possible_files:
    if os.path.exists(filepath):
        saved_file = filepath
        break

if saved_file:
    size = os.path.getsize(saved_file)
    filename = os.path.basename(saved_file)
    print(f"✅ SUCCESS! Model saved to {output_dir}")
    print(f"📦 Model file: {filename} ({size:,} bytes / {size/1024/1024:.2f} MB)")
else:
    print(f"⚠️ WARNING - No model file found in standard locations")
    print(f"   But other files may have been saved - check below:")

if os.path.exists(output_dir) and os.listdir(output_dir):
    print(f"\n📁 Saved files:")
    for item in sorted(os.listdir(output_dir)):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path):
            file_size = os.path.getsize(item_path)
            print(f"   📄 {item} ({file_size:,} bytes)")
        else:
            print(f"   📁 {item}/")
else:
    print(f"⚠️ Output directory is empty or doesn't exist")

print("\n💡 TIP: Your model is saved to /kaggle/working/ and will appear")
print("   in the 'Output' section when you save this notebook!")

print("\n" + "="*70)
print("🧪 TESTING MODEL WITH SAMPLE PROMPTS")
print("="*70)

print("Using trained model from memory...")
tuned_model = trainer.model
tuned_model.eval()  # Set to evaluation mode

test_prompts = [
    "If your friends jumped off a bridge, would you? my mother's scolding rang out in my head.",
    "After I narrowly stopped my best friend from ending it all that one haunting night, I swore that this would never happen again.",
    "My husband was really hurt when I told him I didn't want kids anymore.",
    "I always wave at the little girl in the window across the street.",
]

print("\nGenerating horror stories...\n")

for i, prompt in enumerate(test_prompts, 1):
    print("="*70)
    print(f"Test {i}/{len(test_prompts)}")
    print("="*70)
    print(f"🎭 PROMPT:\n{prompt}\n")
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(tuned_model.device)
        
        with torch.no_grad():
            outputs = tuned_model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Fix for DynamicCache error
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"👻 GENERATED STORY:\n{generated_text}\n")
    except Exception as e:
        print(f"❌ Generation failed: {e}\n")

print("\n" + "="*70)
print("🎉 ALL DONE!")
print("="*70)
print(f"✅ Trained on {len(clean_dataset)} high-quality horror stories")
print(f"✅ Training time: {training_time/60:.1f} minutes")
print(f"✅ Model saved to: {output_dir}")

total_size = sum(
    os.path.getsize(os.path.join(output_dir, f)) 
    for f in os.listdir(output_dir) 
    if os.path.isfile(os.path.join(output_dir, f))
)
print(f"✅ Total model size: {total_size / 1024 / 1024:.1f} MB")

print("\n📝 NEXT STEPS:")
print("   1. Click 'Save Version' (top right) to save this notebook")
print("   2. Your model will appear in the Output section")
print("   3. Click 'New Dataset' on the output to create a reusable dataset")
print("   4. Use that dataset in other notebooks for inference")
print("\n🔗 SHARING:")
print("   - Make notebook public to share with others")
print("   - Add to your portfolio/resume")
print("   - Link from your GitHub README")
print("\n💡 TO REUSE YOUR MODEL LATER:")
print("   # In a new notebook, add your model dataset as input, then:")
print("   from peft import PeftModel")
print("   base = AutoModelForCausalLM.from_pretrained('microsoft/phi-3-mini-128k-instruct', ...)")
print("   model = PeftModel.from_pretrained(base, '/kaggle/input/your-model-dataset/horror-model')")
print("\n" + "="*70)
