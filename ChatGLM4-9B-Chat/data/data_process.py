import json

data_path = '/root/autodl-tmp/ChatGLM4-9B-Chat/data/output3.json'

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)



messages = []
for item in data:
    item['instruction'] = "Please provide analysis in the following format: 1. Basic Information: - Summarize the product details and overall customer sentiment - Highlight key features mentioned in reviews. 2. Personal Budget Considerations:- Analyze any price-related comments- Evaluate value for money based on reviews. 3. Needs Assessment:- Identify common use cases from reviews- Analyze how well the product meets customer needs- Point out any limitations or concerns. 4. Preference Analysis:- Discuss style and design aspects- Analyze user satisfaction patterns- Provide recommendations for potential buyers. Please provide detailed analysis for each section to help potential buyers make informed decisions."
    input_str = {
        "product_name": item['input']['product_name'],
        "review_count": str(item['input']['review_count']),  # 转成字符串
        "average_rating": str(item['input']['average_rating']),  # 转成字符串
        "reviewText": item['input']['reviewText']
    }
    item['input'] = str(input_str)
    messages.append(item)


output_path = '/root/autodl-tmp/ChatGLM4-9B-Chat/data/finetune_data_glm_4.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(messages, f, ensure_ascii=False, indent=4)

print(f"Data has been saved to {output_path}")