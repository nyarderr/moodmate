
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


## load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen1.5-0.5B",
    trust_remote_code=True
)

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
    "emotion_detector_peft_model",
    torch_dtype=torch.float16,
    device_map="auto",
)





def EmotionDetector(text):

    final_text = {
        f"Instrcution: Identify the emotion of the following text.\n"
        f"Text:{text}\n"
        f"Emotion:"
    }
    tokenized_text = model.tokenizer(final_text, return_tensors="pt", padding=True)
    output = model.model.generate(**tokenized_text, max_new_tokens=10)
    decoded_output = model.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    emotion = decoded_output.split("Emotion:")[-1].strip()
    return emotion