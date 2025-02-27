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


prompt = '我有一个医学数字人产品，针对需要对下级医院和医生进行知识培训及分享的医学教授和专家们。医学专家由于专业不对口、事务繁忙等原因无法将自己的知识经验等做成视频分享给其他医生学习，这款产品可以让专家们非常方便地生成以自己形象出镜的知识科普和慕课视频，从而达到快速分享知识的效果。请将我的产品软植入福尔摩斯和华生的故事，不要太明显。19世纪，医生华生协助福尔摩斯探案（需要一个准确的凶杀案描述），在尸检环节由于技术限制华生无法解决，福尔摩斯掏出秘密武器——我们的产品，播放了来自21世纪医生的数字人视频教给华生知识，最终悬案顺利破解。故事需要跌宕起伏的情节和生动的对话。请根据剧情提示写1500字以上的文章。产品特点是只需要上传10秒的本人出镜视频和文案讲稿，就可以生成自己的虚拟形象，表情口型动作都到位，无需真人拍摄。'
chat = [{'role':'user', 'content':prompt}]
response_content = chatbot(chat, max_new_tokens=8192, max_length=8192)[0]['generated_text'][-1]['content']

print('问：', prompt)
print('答：', response_content)
print('长度', len(response_content))

