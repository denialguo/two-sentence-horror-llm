# Install libraries
print("📦 Installing libraries...")
!pip install -q transformers accelerate bitsandbytes peft

# Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")

# Configuration - UPDATE THIS PATH!
MODEL_PATH = "/kaggle/input/two-sentence-horror-model-mistral-7b/"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Load model
print("\n🤖 Loading model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base, MODEL_PATH)
model.eval()
print("✅ Model loaded!\n")

# Generate function
def generate_story(topic=""):
    if topic:
        prompt = f"<s>[INST] Write a creative and chilling two-sentence horror story about {topic}. [/INST]"
    else:
        prompt = "<s>[INST] Write a creative and chilling two-sentence horror story. [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.75,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "[/INST]" in result:
        return result.split("[/INST]")[-1].strip()
    return result

# Test!
print("="*70)
print("👻 GENERATING HORROR STORIES")
print("="*70)

for i in range(5):
    story = generate_story()
    print(f"\n{i+1}. {story}")

print("\n" + "="*70)
print("📝 TOPIC-SPECIFIC STORIES")
print("="*70)

topics = ["mirrors", "a phone call", "being home alone", "a child's toy"]
for topic in topics:
    story = generate_story(topic)
    print(f"\n[{topic}] {story}")

print("\n" + "="*70)
