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
            (5,), device=noise.device
        ).long()

# 第一步：对原始张量进行加噪
noisy_tensor = scheduler.add_noise(original_tensor, noise, noising_timesteps)

# 第二步：进行去噪
prev_samples = []
for i in range(original_tensor.shape[0]):
    # 假设模型输出预测的是噪声（与实际噪声相同）
    model_output = noise[i]  # 模型预测的噪声

    # 使用调度器的 step 函数还原去噪
    step_output = scheduler.step(
        model_output=model_output.unsqueeze(0),  # 输入单个样本
        timestep=noising_timesteps[i],  # 针对每个样本的时间步
        sample=noisy_tensor[i].unsqueeze(0)  # 输入单个样本
    )

    # 获取去噪后的样本
    prev_samples.append(step_output["prev_sample"].squeeze(0))

# 合并去噪后的样本
prev_samples = torch.stack(prev_samples)

# 第三步：计算还原后的样本与原始样本之间的误差
difference = torch.mean((original_tensor - prev_samples) ** 2)

print("Mean Squared Error between original and restored tensor:", difference.item())
