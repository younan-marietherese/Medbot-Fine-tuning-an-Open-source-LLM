# MedBot: Fine-Tuning LLMs for Medical Q&A
Ranked Top 2 in the LLM Fine-Tuning Challenge (Zaka AI – August 2025)
This project focuses on building MedBot, a medical question-answering chatbot, by fine-tuning large language models using LoRA and 4-bit quantization with Hugging Face Transformers.

## Project Summary
The goal was to fine-tune a large language model (LLM) on a medical dataset to answer patient-related questions efficiently — all while working under tight GPU and budget constraints.

We fine-tuned:

- GPT-NeoX-20B (on A100 GPU with LoRA + 4-bit quantization)

- Attempted fallback training on GPT-Neo 2.7B (training crashed at 0.87/1 epoch due to CUDA error)

## Tech Stack
Hugging Face Transformers & PEFT

BitsAndBytes (4-bit quantization)

LoRA (Low-Rank Adaptation)

PyTorch

Google Colab / RunPod (A100 GPU)

## Dataset
https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc

Used as an instruction-response dataset for medical Q&A.

## Training Highlights
4-bit quantized GPT-NeoX-20B model trained with LoRA

Trained for 1 epoch (~2.5 hours) with gradient accumulation

Only 0.08% of model parameters were updated

Handled and debugged inference issues and GPU crashes

Full pipeline: dataset loading → model setup → training → saving → inference
