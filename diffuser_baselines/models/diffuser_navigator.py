import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionNavigator(nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 num_attn_heads=8,
                 diffusion_timesteps=1000,
                 use_instruction=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_instruction = use_instruction

        # 编码器: 将 action 映射到 embedding space
        self.action_encoder = nn.Embedding(1000, embedding_dim)  # 假设动作空间有1000个离散动作

        # Position Encoding (可选)
        self.position_encoding = nn.Parameter(torch.randn(6, embedding_dim))

        # Denoising Transformer
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_attn_heads,
            num_encoder_layers=6,
            num_decoder_layers=6
        )

        # 扩散调度器
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )

        # 输出层 (将 embedding space 转回到动作空间)
        self.action_decoder = nn.Linear(embedding_dim, 1)

    def encode_inputs(self, rgb_features, depth_features, instructions, gt_actions):
        
        # 将 rgb_features 和 depth_features 转换为 embedding
        rgb_tokens = einops.rearrange(rgb_features, 'b s c h w -> b s (h w) c')
        depth_tokens = einops.rearrange(depth_features, 'b s c h w -> b s (h w) c')

        # 对 gt_actions 进行编码
        gt_action_tokens = self.action_encoder(gt_actions)

        # 将 instructions 线性映射到 embedding space
        instruction_tokens = nn.Linear(instructions.size(-1), self.embedding_dim)(instructions)

        # 合并所有 tokens
        tokens = torch.cat([
            rgb_tokens, depth_tokens, instruction_tokens.unsqueeze(2), gt_action_tokens.unsqueeze(2)
        ], dim=2)

        # 加入位置编码
        tokens += self.position_encoding.unsqueeze(0).unsqueeze(2)

        return tokens

    def forward(self, rgb_features, depth_features, instructions, gt_actions):

        B, S, _ = gt_actions.shape

        # 编码输入
        tokens = self.encode_inputs(rgb_features, depth_features, instructions, gt_actions)

        # 添加噪声到动作 tokens
        timesteps = torch.randint(0, self.position_noise_scheduler.config.num_train_timesteps, (B,), device=tokens.device).long()
        noisy_tokens = self.position_noise_scheduler.add_noise(gt_actions, torch.randn_like(gt_actions), timesteps)

        # 使用 Transformer 进行去噪预测
        denoised_tokens = self.transformer(tokens, noisy_tokens)

        # 解码动作
        predicted_actions = self.action_decoder(denoised_tokens)

        # 计算损失
        loss = F.mse_loss(predicted_actions, noisy_tokens)
        return loss

    def sample(self, rgb_features, depth_features, instructions):
        B, S, _ = rgb_features.shape[:3]
        tokens = self.encode_inputs(rgb_features, depth_features, instructions, torch.zeros(B, S).long().to(rgb_features.device))

        # 逐步去噪
        self.position_noise_scheduler.set_timesteps(self.position_noise_scheduler.config.num_train_timesteps)
        timesteps = self.position_noise_scheduler.timesteps
        for t in reversed(timesteps):
            denoised_tokens = self.transformer(tokens, tokens)
            tokens = self.position_noise_scheduler.step(denoised_tokens, t, tokens).prev_sample

        predicted_actions = self.action_decoder(tokens)
        return predicted_actions
