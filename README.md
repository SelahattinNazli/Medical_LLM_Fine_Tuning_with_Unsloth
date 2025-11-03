# Medical LLM Fine-Tuning with Unsloth

This project fine-tunes the **Llama 3.2 3B Instruct** model using **[Unsloth](https://github.com/unslothai/unsloth)** for specialized medical question-answering.  
The goal is to create a **medical assistant model** that can answer professional health-related questions with accuracy and clarity — without using any paid APIs.

---

## Project Overview

Large Language Models (LLMs) such as **Llama 3** are powerful general-purpose models.  
However, when fine-tuned on domain-specific datasets (like medicine), they can achieve **specialized reasoning and terminology fluency**.

In this project, we:
- Use **Unsloth** for fast and efficient LoRA fine-tuning (optimized for 4-bit quantization).
- Train on **Medical Meadow Flashcards** dataset.
- Produce a **custom medical assistant model** that responds like a professional doctor.

---

## Model Architecture

| Component | Description |
|------------|-------------|
| **Base Model** | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` |
| **Quantization** | 4-bit (bnb) for memory-efficient fine-tuning |
| **Fine-Tuning Technique** | LoRA (Low-Rank Adaptation) |
| **Optimizer** | AdamW 8-bit |
| **Precision** | Auto (FP16/BF16) |
| **Training Framework** | UnslothTrainer |
| **GPU** | T4 / A100 compatible |

---

## Dataset

We use the open medical dataset:

**`medalpaca/medical_meadow_medical_flashcards`**

Each sample includes:
- `input`: a medical question  
- `output`: a professional answer  

We convert them into an **Alpaca-style instruction format** for fine-tuning:

```text
You are an expert doctor in the field of medicine.
Answer the following medical question professionally and accurately.

### Question:
<medical question>

### Medical Answer:
<expert medical answer>
```
## Training Configuration
```bash
Parameter	Value
train_size	10% of dataset
max_seq_length	1024
batch_size	6
gradient_accumulation_steps	2
num_train_epochs	1
learning_rate	5e-5
scheduler	Cosine
warmup_ratio	0.05
LoRA rank	64
LoRA alpha	128
LoRA dropout	0
use_rslora	 Yes
```
## Training Script
The main fine-tuning process is handled by:

```bash
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    args = UnslothTrainingArguments(
        per_device_train_batch_size = 6,
        num_train_epochs = 1,
        learning_rate = 5e-5,
        output_dir = "/content/drive/MyDrive/unsloth_medical_assistant",
        report_to = "none",
    ),
)
trainer.train()
```
### Inference Example
After training, we convert the model for inference:
```bash
FastLanguageModel.for_inference(model)

test_prompt = """You are an expert doctor in the field of medicine.
Answer the following medical question professionally and accurately.

### Question:
What are the common symptoms of diabetes?

### Medical Answer:
"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
## Example Output:

```bash
The most common symptoms of diabetes include excessive urination (polyuria),
increased thirst (polydipsia), fatigue, blurred vision, slow wound healing,
and weight loss.
```
## Environment Setup
```bash
import os

IN_COLAB = 'COLAB_GPU' in os.environ
if IN_COLAB:
    print("Installing dependencies for Colab environment...")
    !pip install -q unsloth "xformers<0.0.30" trl peft bitsandbytes
else:
    print("Local environment detected")
    !pip install -q unsloth
```
```bash
!pip install -q datasets transformers accelerate
```
## Results
Metric	Observation
```bash
Training Time	~25 min (on T4)
Loss	↓ Converging smoothly
Output Quality	Coherent, domain-consistent, medically accurate
Tokens per Response	~150–200
```

## Deployment

The fine-tuned model can be served via **FastAPI**. After cloning the repository and installing the requirements, start the server with:

```bash
uvicorn app:app --reload
```

## Key Takeaways
✅ Demonstrates how to fine-tune a domain-specific LLM with Unsloth

✅ Fully local and API-free — all training done with open tools

✅ Produces contextually accurate, professional medical answers

✅ Ready for downstream tasks like medical chatbots, tutoring, or QA evaluation

## Future Improvements
Add evaluation metrics (ROUGE, cosine similarity, BERTScore)

Extend dataset coverage (e.g. PubMedQA, MedMCQA)

Fine-tune on multi-turn dialogues

Deploy model with Gradio or FastAPI
