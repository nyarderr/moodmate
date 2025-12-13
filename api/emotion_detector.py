import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LORA_PATH = "../models/qwen_lora_emotions"

## load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)

## load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)


## load peft model
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load LoRA adapter


model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()


def detect_emotion(text: str) -> str:
    prompt = (
        "Instrcution: Identify the emotion of the following text.\n"
        f"Text: {text}\n"
        "Emotion:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    emotion = decoded.split("Emotion:")[-1].strip().split()[0]

    return emotion
