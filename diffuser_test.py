import torch
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# 假设 DDPMScheduler 已经导入
# 初始化 DDPMScheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 创建一个形状为 (5, 6, 256) 的张量
original_tensor = torch.randn(5, 6, 256)

# 生成随机噪声，形状和 original_tensor 一样
noise = torch.randn_like(original_tensor)

# 选择一个时间步（比如从最大的时间步开始，1000步的最后一步）
timestep = torch.tensor(999)

# 将原始张量加噪
noisy_tensor = scheduler.add_noise(original_tensor, noise, timestep)

# 使用模型的去噪步骤
# 模拟模型输出为纯噪声
model_output = noise

# 使用调度器的 step 函数来逐步还原张量
# step 函数需要输入 model_output（模型的输出）和 noisy_tensor（加噪后的输入）
prev_sample = scheduler.step(
    model_output=model_output,
    timestep=timestep,
    sample=noisy_tensor
).prev_sample

# 比较还原后的张量和原始张量
# 计算还原张量与原始张量的差异
difference = torch.mean((original_tensor - prev_sample) ** 2)
print("Mean Squared Error between original and restored tensor:", difference.item())
