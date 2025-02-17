from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
# 在modelscope上下载GLM模型到本地目录下
model_dir = snapshot_download("ZhipuAI/glm-4-9b-chat", cache_dir="./", revision="master")


tokenizer = AutoTokenizer.from_pretrained("./ZhipuAI/glm-4-9b-chat/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./ZhipuAI/glm-4-9b-chat/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
