from peft import PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, pipeline, AutoModelForCausalLM
import torch

model_path = '/root/autodl-tmp/ChatGLM4-9B-Chat/glm-4-9b-chat'
peft_path = '/root/autodl-tmp/ChatGLM4-9B-Chat/checkpoint'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
# model = PeftModel.from_pretrained(model, peft_path)
# model = model.merge_and_unload()


chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer)


prompt = 'As a shopping decision assistant, generate a detailed shopping recommendation report based on the basic information of the item, including the item name, number of reviews, and average rating. The report should include basic information analysis, budget considerations, needs assessment and preference analysis.'
chat = [{'role':'user', 'content':prompt}]
response_content = chatbot(chat, max_new_tokens=8192, max_length=8192)[0]['generated_text'][-1]['content']

print('问：', prompt)
print('答：', response_content)


