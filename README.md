# 🎃 Two-Sentence Horror Story Generator

*Fine-tuned LLM that generates creepy two-sentence horror stories, trained on 
top-rated content from r/TwoSentenceHorror*

## Overview
A parameter-efficient fine-tuning (PEFT) project that adapts Microsoft's Phi-3 
model to generate creative horror micro-fiction. The model learns narrative 
structure and suspense-building techniques from thousands of highly-rated 
Reddit horror stories.

## Tech Stack
- **Model**: Microsoft Phi-3-mini-128k-instruct
- **Training**: LoRA (Low-Rank Adaptation) via HuggingFace PEFT
- **Optimization**: 4-bit quantization (bitsandbytes), QLoRA
- **Framework**: Transformers, TRL (Transformer Reinforcement Learning)
- **Dataset**: 3,185 of the top stories from r/TwoSentenceHorror, as well as {amount} stories from the top of every month. 

## Key Features
- ✅ Memory-efficient training using LoRA (trains only 0.1% of parameters)
- ✅ Quality-filtered dataset (>85% upvote ratio)
- ✅ 4-bit quantization for consumer GPU compatibility
- ✅ Generates coherent, creepy micro-narratives
