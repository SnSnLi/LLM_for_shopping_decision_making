import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from peft import PeftModel, get_peft_model
from PyPDF2 import PdfReader

# 将PDF文件转换为文本
def pdf_to_text(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# PDF文件路径
pdf_path = 'autodl-tmp/inference/人工智能标准化白皮书 2018版.pdf'
# 转换PDF为文本
knowledge_base_text = pdf_to_text(pdf_path)

# 初始化RAG的tokenizer和retriever
rag_tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
rag_retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', index_name='exact', use_dummy_dataset=False)

# 将知识库文本分割成段落，并添加到RAG的索引中
paragraphs = knowledge_base_text.split('\n\n')  # 假设每个段落由两个换行符分隔
for i, paragraph in enumerate(paragraphs):
    rag_retriever.add_document(i, paragraph)

# 初始化RAG模型
rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq', retriever=rag_retriever)

# 加载微调后的GLM-4模型
glm4_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/output_path", trust_remote_code=True)
glm4_model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/output_path", trust_remote_code=True)

# 加载RAG模型和PEFT模型
model_path = '/root/autodl-tmp/pretrained_model/Qwen2-7B-Instruct'
peft_path = '/root/autodl-tmp/LLaMA-Factory/saves/llama3-8b/lora/sft_epoch20'
rag_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
rag_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
rag_model = PeftModel.from_pretrained(model_path, peft_path)
rag_model = rag_model.merge_and_unload()

# 使用RAG模型生成上下文
def generate_context(question, rag_tokenizer, rag_model):
    input_ids = rag_tokenizer(question, return_tensors="pt").input_ids
    outputs = rag_model.generate(input_ids, max_length=150)
    return rag_tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用GLM-4模型生成回答
def generate_answer(context, question, glm4_tokenizer, glm4_model):
    input_ids = glm4_tokenizer(question, context, return_tensors="pt").input_ids
    outputs = glm4_model.generate(input_ids, max_length=150)
    return glm4_tokenizer.decode(outputs[0], skip_special_tokens=True)

# 创建一个生成器管道
chatbot = pipeline('text-generation', model=glm4_model, tokenizer=glm4_tokenizer)

# 用户输入
prompt = input('问：')
chat = [{'role':'user', 'content':prompt}]

# 使用GLM-4模型生成回答
response_content = chatbot(chat, max_new_tokens=512)[0]['generated_text'][-1]['content']
# 输出GLM-4模型的回答
print('问：', prompt)
print('答：', response_content)

# 使用RAG模型对生成的回答进行检索增强
context = generate_context(response_content, rag_tokenizer, rag_model)

# 输出RAG检索增强后的上下文
print('检索增强后的上下文：', context)

# 使用RAG模型生成增强后的回答
enhanced_answer = generate_answer(context, prompt, rag_tokenizer, rag_model)

# 输出增强后的回答
print('增强后的回答：', enhanced_answer)

