import torch
from transformers import pipeline
print(torch.cuda.is_available())
chat = [
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipe = pipeline("text-generation", "mistralai/Mistral-7B-Instruct-v0.3", torch_dtype=torch.bfloat16, device_map="auto")
response = pipe(chat, max_new_tokens=512)
print(response[0]['generated_text'][-1]['content'])

chat = response[0]['generated_text']
chat.append(
    {"role": "user", "content": "Wait, what's so wild about soup cans?"}
)
response = pipe(chat, max_new_tokens=512)
print(response[0]['generated_text'][-1]['content'])