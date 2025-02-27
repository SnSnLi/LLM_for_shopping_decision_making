import torch

# 假设在某个点，你发现你的模型不再需要之前分配的内存
# 你可以调用empty_cache()来释放那些内存
torch.cuda.empty_cache()

# 继续你的代码，比如继续训练模型
