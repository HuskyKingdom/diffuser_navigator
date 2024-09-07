from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch

# 初始化调度器
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 创建一个形状为 (5, 6, 256) 的张量，表示原始数据
original_tensor = torch.randn(5, 6, 256)

# 生成与原始张量形状一致的随机噪声
noise = torch.randn_like(original_tensor)

# 选择一个时间步，比如 timestep = 999
noising_timesteps = torch.randint(
            0,
            1000,
            (len(noise),), device=noise.device
        ).long()

# 第一步：对原始张量进行加噪
noisy_tensor = scheduler.add_noise(original_tensor, noise, noising_timesteps)

# 第二步：进行去噪
# 假设模型输出预测的是噪声（与实际噪声相同）
model_output = noise

# 使用调度器的 step 函数还原去噪
step_output = scheduler.step(
    model_output=model_output,
    timestep=noising_timesteps,
    sample=torch.tensor(noisy_tensor)
)

# 获取去噪后的样本
prev_sample = step_output["prev_sample"]

# 第三步：计算还原后的样本与原始样本之间的误差
difference = torch.mean((original_tensor - prev_sample) ** 2)
print("Mean Squared Error between original and restored tensor:", difference.item())
