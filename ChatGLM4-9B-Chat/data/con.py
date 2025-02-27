import json

def convert_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = []
    for item in data:
        new_item = {
            "instruction": "Please evaluate the product information comprehensively and give reasonable purchase suggestions",
            "input": item["input"],
            "output": item["output"]
        }
        new_data.append(new_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

# 使用
convert_format('output3.json', 'output3.json')