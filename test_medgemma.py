from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="google/medgemma-27b-text-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful medical assistant."
    },
    {
        "role": "user",
        "content": "How do you differentiate bacterial from viral pneumonia?"
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])