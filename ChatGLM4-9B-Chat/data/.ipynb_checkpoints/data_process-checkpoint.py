import json

data_path = '/root/autodl-tmp/ChatGLM4-9B-Chat-Weight/data/generated_ads1.json'

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)



messages = []
for item in data:
    item['instruction'] = item['input']
    item['input'] = ''
    messages.append(item)


output_path = '/root/autodl-tmp/ChatGLM4-9B-Chat-Weight/data/finetune_data_glm_4.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(messages, f, ensure_ascii=False, indent=4)

print(f"Data has been saved to {output_path}")