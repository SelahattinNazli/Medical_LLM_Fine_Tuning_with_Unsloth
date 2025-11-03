from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch

# ============================================================
# 1️⃣ Load fine-tuned model from your saved directory
# ============================================================
model_path = "/content/drive/MyDrive/unsloth_medical_final"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=1024,
    dtype=None,  # Auto select FP16 or BF16
    load_in_4bit=True,  # Memory efficient
)

FastLanguageModel.for_inference(model)

# ============================================================
# 2️⃣ Define FastAPI app
# ============================================================
app = FastAPI(title="Fine-Tuned Medical LLM API", version="1.0")


class Question(BaseModel):
    question: str


# ============================================================
# 3️⃣ Define generation endpoint
# ============================================================
@app.post("/generate")
async def generate_answer(item: Question):
    prompt = f"""You are an expert doctor in the field of medicine.
Answer the following medical question professionally and accurately.

### Question:
{item.question}

### Medical Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"question": item.question, "answer": answer}


# ============================================================
# 4️⃣ Health check endpoint
# ============================================================
@app.get("/")
def home():
    return {"message": "Fine-tuned Unsloth Medical Model API is running!"}
