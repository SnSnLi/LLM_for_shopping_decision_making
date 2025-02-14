import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取JSON文件
    with open(origin_path, "r") as file:
        data = json.load(file)  # 整个文件作为一个JSON对象读取
        if isinstance(data, list):  # 如果是JSON数组
            for item in data:
                message = {
                    "instruction": "作为一个购物决策助手，请根据商品的基本信息生成详细的购物建议报告。",
                    "input": f"商品信息：\n- 商品名称：{item['input']['product_name']}\n- 评价数量：{item['input']['review_count']}\n- 平均评分：{item['input']['average_rating']}",
                    "output": item['output']
                }
                messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 2048  # 增加长度以适应详细的购物建议报告
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|system|>\n作为一个购物决策助手，请根据商品的基本信息（包括商品名称、评价数量和平均评分），生成一份详细的购物建议报告。报告应该包含基本信息分析、预算考虑、需求评估和偏好分析等方面。<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 模型加载
model_dir = snapshot_download("ZhipuAI/glm-4-9b-chat", cache_dir="./", revision="master")
tokenizer = AutoTokenizer.from_pretrained("./ZhipuAI/glm-4-9b-chat/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./ZhipuAI/glm-4-9b-chat/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads()

# 数据处理
train_dataset_path = "output.json"  # 你的训练集文件
test_dataset_path = "output1.json"  # 你的测试集文件

train_jsonl_new_path = "new_train.jsonl"
test_jsonl_new_path = "new_test.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 训练集处理
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "activation_func", "dense_4h_to_h"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

# 训练参数
args = TrainingArguments(
    output_dir="./output/GLM4-9b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 测试
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    print(f"Test {index + 1}:")
    print(f"Input: {input_value}")
    print(f"Prediction: {response}")
    print("-" * 50)