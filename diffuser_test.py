from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch

# 初始化调度器
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 创建一个形状为 (5, 6, 256) 的张量，表示原始数据
original_tensor = torch.randn(1, 6, 256)

# 生成与原始张量形状一致的纯随机噪声
noise = torch.randn_like(original_tensor)

# 设定从 T=999 开始
T = 999
noising_timesteps = torch.full((5,), T, device=noise.device, dtype=torch.long)

# 第一步：对原始张量进行加纯噪声
noisy_tensor = scheduler.add_noise(original_tensor, noise, noising_timesteps)

# 第二步：从 T=999 到 T=0 逐步去噪
for t in range(T, -1, -1):  # 从999降到0
    prev_samples = []
    for i in range(original_tensor.shape[0]):
        # 模型输出假设为噪声（与实际噪声相同）
        model_output = noise  # 模型预测的噪声

        # 使用调度器的 step 函数逐步去噪，sample 是上一轮去噪结果
        step_output = scheduler.step(
            model_output=model_output,  # 输入单个样本
            timestep=torch.tensor([t], device=noisy_tensor.device),  # 当前时间步
            sample=noisy_tensor[i]  # 上一轮去噪后的结果作为输入
        )

        # 获取去噪后的样本
        prev_samples.append(step_output["prev_sample"].squeeze(0))

    # 更新 noisy_tensor 为当前的去噪结果，用于下一轮去噪
    noisy_tensor = torch.stack(prev_samples)
    print(noisy_tensor.shape)

# 第三步：计算还原后的样本与原始样本之间的 L1 距离
l1_difference = torch.mean(torch.abs(original_tensor - noisy_tensor))

print("L1 Distance between original and restored tensor:", l1_difference.item())
