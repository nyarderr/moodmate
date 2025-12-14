# api/response_generator.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)

model.eval()


def generate_support_message(text: str, emotion: str) -> str:
    prompt = (
        f"The user is feeling {emotion}.\n"
        f'User message: "{text}"\n\n'
        "Write a short, empathetic, and supportive response. "
        "Do not give medical advice."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=80, do_sample=True, temperature=0.7
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in decoded:
        response = decoded.split("Assistant:")[-1].split("User message:")[0].strip()
    else:
        response = decoded.strip()

    return response
