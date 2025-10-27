import os, torch
os.environ["HF_HUB_CACHE"] = "/data/not_backed_up/amukundan/hf_models"
from transformers import pipeline, BitsAndBytesConfig

model_id = "google/medgemma-27b-text-it"

bnb8 = BitsAndBytesConfig(load_in_8bit=True)

bnb8 = BitsAndBytesConfig(load_in_8bit=True)

pipe = pipeline(
    task="text-generation",
    model="google/medgemma-27b-text-it",
    tokenizer="google/medgemma-27b-text-it",
    model_kwargs={
        "quantization_config": bnb8,
        "device_map": "auto",
        "dtype": torch.bfloat16,
    },
)
messages = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "How do you differentiate bacterial from viral pneumonia?"}
]
out = pipe(messages)
print(out[0]["generated_text"][-1]["content"])