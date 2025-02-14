import os
import random

import numpy as np
import torch

# 常见生成随机数的播种方法，只要播种相同就可以保证接下来生成的随机数相同
seed = 1
os.environ["PYTHONHASHSEED"] = str(seed)
print("hash of s is", hash("abc"))
# 以上可以设置Python的Hash种子，保证Hash的结果

np.random.seed(seed)
np.random.rand(5)

random.seed(seed)
random.random()

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.rand(5)

# 置为True的话，cudnn的卷积算法将是确定的
torch.backends.cudnn.deterministic = True
# 置为True的话，cudnn自动寻找最适合当前配置的高效算法，增加运行效率，置为False可以去除cudnn计算的随机性，因为不同算法的精度是不同的
torch.backends.cudnn.benchmark = False
