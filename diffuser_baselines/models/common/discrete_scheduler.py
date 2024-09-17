from diffusers.schedulers.scheduling_utils import SchedulerMixin
import torch
import torch.nn.functional as F

class DiscreteDDPMScheduler(SchedulerMixin):
    def __init__(self, num_train_timesteps, num_classes, device, initial_noise_rate=0.0, final_noise_rate=0.5):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps  # 扩散过程的时间步数
        self.num_classes = num_classes  # 动作类别数
        self.device = device

        self.initial_noise_rate = initial_noise_rate  # 初始噪声率
        self.final_noise_rate = final_noise_rate      # 最终噪声率

        # 定义每个时间步的转移概率矩阵
        self.transition_probs = self._create_transition_probs()

        # 预先计算累积转移概率，用于加速采样
        self.cumulative_transition_probs = torch.cumsum(self.transition_probs, dim=-1)
    
    def _create_transition_probs(self):
        # 生成噪声率序列
        noise_rates = torch.linspace(
            self.initial_noise_rate, self.final_noise_rate, self.num_train_timesteps, device=self.device
        ).view(-1, 1, 1)  # (num_train_timesteps, 1, 1)

        # 创建对角矩阵和非对角矩阵
        eye = torch.eye(self.num_classes, device=self.device)  # (num_classes, num_classes)
        off_diag = 1 - eye  # 非对角线元素为 1

        # 计算转移概率矩阵，利用广播机制
        transition_probs = (1 - noise_rates) * eye + (noise_rates / (self.num_classes - 1)) * off_diag  # (num_train_timesteps, num_classes, num_classes)

        return transition_probs

    def add_noise(self, original_samples, timesteps):
        """
        根据转移概率矩阵，将原始动作序列转换为加噪后的动作序列。

        original_samples: (batch_size, seq_len) 的张量，元素取值为 [0, num_classes-1]
        timesteps: (batch_size,) 的张量，表示每个样本对应的时间步
        """
        batch_size, seq_len = original_samples.shape

        # 获取每个样本对应的转移概率矩阵，形状为 (batch_size, num_classes, num_classes)
        transition_probs_per_sample = self.transition_probs[timesteps]

        # 获取原始样本的类别，形状为 (batch_size, seq_len)
        current_classes = original_samples

        # 利用高级索引获取每个位置的转移概率，形状为 (batch_size, seq_len, num_classes)
        probs = transition_probs_per_sample[
            torch.arange(batch_size)[:, None],  # (batch_size, 1)
            current_classes  # (batch_size, seq_len)
        ]

        # 进行采样，形状为 (batch_size, seq_len)
        next_classes = torch.multinomial(probs.view(-1, self.num_classes), num_samples=1).view(batch_size, seq_len)

        return next_classes

    def step(self, model_output, timestep, sample):
        """
        根据模型的输出概率分布，预测下一个时间步的动作序列。

        model_output: (batch_size, seq_len, num_classes) 的张量，模型输出的类别概率分布（未经过 softmax）
        timestep: 当前时间步（整数）
        sample: 当前的样本（动作序列），形状为 (batch_size, seq_len)
        """
        # 将模型输出转换为概率分布，形状为 (batch_size, seq_len, num_classes)
        probs = F.softmax(model_output, dim=-1)

        # 进行采样，形状为 (batch_size, seq_len)
        predicted_samples = torch.multinomial(probs.view(-1, self.num_classes), num_samples=1).view(sample.shape)

        return {"prev_sample": predicted_samples}
    
    def set_timesteps(self, num_inference_steps):
        """
        设置推理过程中的时间步数。

        num_inference_steps: 推理过程的总时间步数
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device
        )

    def __len__(self):
        return self.num_train_timesteps
