from transformers import *
from peft import *
import torch

model_path = '/root/autodl-tmp/ChatGLM4-9B-Chat/glm-4-9b-chat'
peft_path = '/root/autodl-tmp/ChatGLM4-9B-Chat/checkpoint'
device = torch.device('cuda')
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model = PeftModel.from_pretrained(model, peft_path)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
model = model.to(device)


chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer)

chat = [{'role': 'user', 'content': '生成数字人的软广'}]
print(chatbot(chat, max_new_tokens=512))