from diffusers.schedulers.scheduling_utils import SchedulerMixin
import torch
import torch.nn.functional as F

class DiscreteDDPMScheduler(SchedulerMixin):
    def __init__(self, num_train_timesteps, num_classes, device):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps  # 扩散过程的时间步数
        self.num_classes = num_classes  # 动作类别数
        self.device = device

        # 定义每个时间步的转移概率矩阵
        # 这里我们使用均匀转移概率，即每个类别有相同的概率转移到其他类别
        self.transition_probs = self._create_transition_probs()

        # 预先计算累积转移概率，用于加速采样
        self.cumulative_transition_probs = torch.cumsum(self.transition_probs, dim=-1)
    
    def _create_transition_probs(self):
        # 创建一个形状为 (num_train_timesteps, num_classes, num_classes) 的转移概率矩阵
        # transition_probs[t, i, j] 表示在时间 t，从类别 i 转移到类别 j 的概率

        # 我们定义一个简单的线性调度，从初始噪声率到最大噪声率
        initial_noise_rate = 0.0  # 初始噪声率
        final_noise_rate = 0.1    # 最终噪声率，可以根据需要调整

        noise_rates = torch.linspace(initial_noise_rate, final_noise_rate, self.num_train_timesteps)

        transition_probs = torch.zeros((self.num_train_timesteps, self.num_classes, self.num_classes), device=self.device)

        for t in range(self.num_train_timesteps):
            noise_rate = noise_rates[t]
            # 对角线元素表示保持当前类别的概率
            transition_probs[t] = (1 - noise_rate) * torch.eye(self.num_classes, device=self.device)
            # 非对角线元素表示转移到其他类别的概率（均匀分布）
            transition_probs[t] += noise_rate / (self.num_classes - 1) * (1 - torch.eye(self.num_classes, device=self.device))

        return transition_probs

    def add_noise(self, original_samples, timesteps):
        """
        根据转移概率矩阵，将原始动作序列转换为加噪后的动作序列。

        original_samples: (batch_size, seq_len) 的张量，元素取值为 [0, num_classes-1]
        timesteps: (batch_size,) 的张量，表示每个样本对应的时间步
        """
        batch_size, seq_len = original_samples.shape
        noised_samples = original_samples.clone()

        for i in range(batch_size):
            t = timesteps[i]
            transition_prob = self.transition_probs[t]  # (num_classes, num_classes)
            for j in range(seq_len):
                current_class = original_samples[i, j]
                # 根据转移概率采样下一个类别
                probs = transition_prob[current_class]  # (num_classes,)
                next_class = torch.multinomial(probs, num_samples=1)
                noised_samples[i, j] = next_class

        return noised_samples

    def step(self, model_output, timestep, sample):
        """
        根据模型的输出概率分布，预测下一个时间步的动作序列。

        model_output: (batch_size, seq_len, num_classes) 的张量，模型输出的类别概率分布（未经过 softmax）
        timestep: 当前时间步（整数）
        sample: 当前的样本（动作序列），形状为 (batch_size, seq_len)
        """
        batch_size, seq_len, num_classes = model_output.shape
        predicted_samples = sample.clone()

        # 将模型输出转换为概率分布
        probs = F.softmax(model_output, dim=-1)  # (batch_size, seq_len, num_classes)

        # 使用类别概率分布进行采样
        for i in range(batch_size):
            for j in range(seq_len):
                predicted_class = torch.multinomial(probs[i, j], num_samples=1)
                predicted_samples[i, j] = predicted_class

        return {"prev_sample": predicted_samples}
    
    def set_timesteps(self, num_inference_steps):
        """
        设置推理过程中的时间步数。

        num_inference_steps: 推理过程的总时间步数
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)

    def __len__(self):
        return self.num_train_timesteps
