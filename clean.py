import torch
import gc

# 清理PyTorch的GPU缓存
torch.cuda.empty_cache()
# 清理Python对象
gc.collect()