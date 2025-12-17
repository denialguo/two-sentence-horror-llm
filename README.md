Two-Sentence Horror Story Generator (Mistral 7B)

This project fine-tunes Mistral 7B Instruct v0.3 to generate short-form horror microfiction using the top 10,000 most upvoted stories from r/TwoSentenceHorror.
Training is performed using LoRA adapters and 4-bit NF4 quantization, enabling efficient fine-tuning on limited hardware.

1. Data Source

The dataset is derived from a full Pushshift subreddit dump, allowing access to historical posts beyond Reddit API limits.

Input Dump: TwoSentenceHorror_submissions.zst

Extraction Script: extract_top_10k.py

Output Dataset: dataset_10k.txt

Each line in the dataset contains:

The post title

The post body

Combined into a single, clean text sample

Extraction Process

The extraction script automatically:

Filters out deleted or removed posts

Sorts all submissions by upvote score

Selects the top 10,000 highest-rated stories

Normalizes punctuation and whitespace

2. Training Notebook Configuration

Training is performed in the llama-2sentencehorror-trainer.ipynb notebook on Kaggle.

Step 3: Load Dataset

The dataset is loaded as a plain text file using Hugging Face Datasets.

⚠️ Important: The path must point to the actual .txt file inside the Kaggle dataset directory.

data_file_path = "/kaggle/input/10k-most-upvoted-two-sentence-horror-2022/dataset_10k.txt"

print(f"Loading from: {data_file_path}")

raw_dataset = load_dataset(
    "text",
    data_files={"train": data_file_path},
    split="train"
)

Step 4: Instruction Formatting

Each story is wrapped using Mistral’s instruction format so the model learns to associate the prompt with the desired output style.

def format_instruction(example):
    story = example["text"].strip()
    return {
        "text": f"<s>[INST] Write a creative and chilling two-sentence horror story. [/INST] {story}</s>"
    }

formatted_dataset = raw_dataset.map(
    format_instruction,
    remove_columns=raw_dataset.column_names
)

3. Training Hyperparameters

Base Model: mistralai/Mistral-7B-Instruct-v0.3

Quantization: 4-bit NF4 (bitsandbytes)

Adapter Method: LoRA

LoRA Configuration

Rank: 8

Alpha: 16

Target Modules:

q_proj

k_proj

v_proj

o_proj

Optimization

Learning Rate: 2e-4

Batch Size: 2 per device

Gradient Accumulation: 2
→ Effective Batch Size: 4

Epochs: 1 (sufficient for 10k high-quality samples)

Optimizer: Paged AdamW (8-bit)

4. Inference Prompt

The model is trained to respond to the following fixed instruction prompt: 
<s>[INST] Write a creative and chilling two-sentence horror story. [/INST]
During inference, the model typically produces:

A grounded setup sentence

Followed by a disturbing or ironic twist

The EOS token (</s>) usually ends generation naturally.
