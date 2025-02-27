from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# 原始模型权重
base_path = '/root/autodl-tmp/pretrained_model/Qwen2-7B-Instruct'
# 训练好的lora权重
peft_path = '/root/autodl-tmp/LLaMA-Factory/saves/llama3-8b/lora/sft_epoch20'


# 加载原始模型的权重
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=True)

# 合并lora参数
model = PeftModel.from_pretrained(model, peft_path)
model = model.merge_and_unload()

# 合并后的权重保存文件路径
save_path = '/root/autodl-tmp/output_path'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print('save successful')

