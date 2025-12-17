Two-Sentence Horror Story Generator (Mistral 7B)

This project fine-tunes a Mistral 7B Instruct model on the top 10,000 most upvoted stories from r/TwoSentenceHorror.

1. Data Source

The dataset is derived from a full Pushshift dump (TwoSentenceHorror_submissions.zst).

Script: extract_top_10k.py

Output: dataset_10k.txt (One story per line, Title and Text combined).

2. Training Notebook Configuration

The training is done in the llama-2sentencehorror-trainer.ipynb notebook on Kaggle.

Step 3: Load Dataset

This loads the dataset text file.
(Note: Ensure the path points to the specific .txt file inside the dataset folder)

# Update this path to where you uploaded your data
data_file_path = f"/kaggle/input/10k-most-upvoted-two-sentence-horror-2022"

print(f"Loading from: {data_file_path}")
raw_dataset = load_dataset("text", data_files={"train": data_file_path}, split="train")


Step 4: Format Instruction

This applies the Mistral 7B instruction format to the dataset.

def format_instruction(example):
    story = example['text'].strip()
    formatted = f"""<s>[INST] Write a creative and chilling two-sentence horror story. [/INST] {story}</s>"""
    return {"text": formatted}

formatted_dataset = raw_dataset.map(format_instruction, remove_columns=raw_dataset.column_names)


3. Training Hyperparameters

Model: mistralai/Mistral-7B-Instruct-v0.3

LoRA Config: Rank 8, Alpha 16, Target Modules: [q_proj, k_proj, v_proj, o_proj]

Quantization: 4-bit NF4

Learning Rate: 2e-4

Batch Size: 2 per device (Effective Batch Size: 4)

Epochs: 1 (Sufficient for 10k items)

4. Inference

The model is trained to respond to the standard prompt:

<s>[INST] Write a creative and chilling two-sentence horror story. [/INST]
4. Inference

This repository provides a LoRA adapter, not a standalone model.
To run inference, you must load the base Mistral 7B Instruct v0.3 model in 4-bit quantization and then apply the adapter using PEFT.

Inference Setup

Requirements

Python 3.10+

transformers

peft

bitsandbytes

torch

pip install transformers peft bitsandbytes accelerate

Example Inference Code
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "path/to/adapter"  # downloaded from Kaggle

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

prompt = "<s>[INST] Write a creative and chilling two-sentence horror story. [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.75,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Output Behavior

The model is trained to generate two-sentence horror micro-fiction

Typical structure:

A grounded setup sentence

A disturbing or ironic twist sentence

The model usually stops naturally due to the </s> EOS token

max_new_tokens=100 is used as a safety cap to prevent runaway generation

Recommended Generation Settings

Temperature: 0.7–0.8

Top-p: 0.9

Max tokens: 100

These values balance creativity with coherence and sentence control.

Hardware Requirements
Training

Performed on Kaggle Free Tier

2× NVIDIA T4 GPUs

LoRA + 4-bit quantization enabled training within limited VRAM

Inference

4-bit quantization allows consumer GPUs

Minimum VRAM: ~6–8 GB

CPU inference is possible but slow and not recommended

Limitations

The model is optimized specifically for two-sentence horror

It may occasionally:

Produce slightly more than two sentences

Merge sentences with semicolons or commas

Not suitable for:

Long-form storytelling

Instruction-following beyond the trained prompt

Factual or non-fiction tasks

Dataset & Attribution

Training data sourced from top-voted posts on r/TwoSentenceHorror

Data extracted from Pushshift subreddit dumps

All content was filtered to remove deleted or removed posts

This project is for research and educational purposes

Future Work

Add explicit sentence-count stopping logic

Experiment with higher LoRA ranks

Compare Mistral vs LLaMA-style instruction formatting

Release a merged, inference-ready checkpoint

License

This project is released under the MIT License.
Base model copyright remains with Mistral AI.
