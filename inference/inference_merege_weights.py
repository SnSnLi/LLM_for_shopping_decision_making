from peft import PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, pipeline, AutoModelForCausalLM


model_path = '/root/autodl-tmp/output_path'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer)


prompt = input('问：')
chat = [{'role':'user', 'content':prompt}]
response_content = chatbot(chat, max_new_tokens=5000)[0]['generated_text'][-1]['content']

print('问：', prompt)
print('答：', response_content)

