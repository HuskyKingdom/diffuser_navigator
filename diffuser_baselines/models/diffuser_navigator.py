import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from habitat_baselines.common.baseline_registry import baseline_registry

from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule,ParallelAttention
from habitat_baselines.rl.ppo.policy import Policy
from diffuser_baselines.models.encoders.instruction_encoder import InstructionEncoder
from diffuser_baselines.models.common.position_encodings import RotaryPositionEncoding,SinusoidalPosEmb


@baseline_registry.register_policy
class DiffusionPolicy(Policy):
    
    def __init__(
        self, config,
        num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps
    ) -> None:
        
        super(Policy, self).__init__()
        self.navigator = DiffusionNavigator(config,num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps)

    def act(observations):

        pass
    

    def build_logits(self,observations):

        out = self.navigator(observations)
    
    @classmethod
    def from_config(
        cls,config, num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps
    ):
        return cls(
            config,num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps
        )




class DiffusionNavigator(nn.Module):

    def __init__(self, config,num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps):

        super(DiffusionNavigator, self).__init__()


        self.embedding_dim = embedding_dim
        self.num_actions = num_actions

        # Encoders
        self.instruction_encoder = InstructionEncoder(config,embedding_dim)
        self.rgb_linear = nn.Linear(16,embedding_dim)
        self.depth_linear = nn.Linear(16,embedding_dim)
        self.action_encoder = nn.Embedding(num_actions, embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        for param in self.action_encoder.parameters(): # freeze action representations
            param.requires_grad = False

        # positional embeddings
        self.pe_layer = RotaryPositionEncoding(embedding_dim)

        # Attention layers
        layer = ParallelAttention(
            num_layers=num_layers,
            d_model=embedding_dim, n_heads=num_attention_heads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.vl_attention = nn.ModuleList([
            layer
            for _ in range(1)
            for _ in range(1)
        ])

        self.cross_attention = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)
        self.self_attention = FFWRelativeSelfAttentionModule(embedding_dim,num_attention_heads,num_layers)

        # Diffusion schedulers
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
        )

        self.n_steps = diffusion_timesteps


    def forward(self, observations, run_inference=False):

        bs = observations["instruction"].size(0)

        
        # tokenlize
        instr_tokens = self.instruction_encoder(observations["instruction"])  # (bs, embedding_dim)
        rgb_tokens = self.rgb_linear(observations["rgb_features"].view(bs,observations["rgb_features"].size(1),-1))  # (bs, 2048, em)
        depth_tokens = self.depth_linear(observations["depth_features"].view(bs,observations["depth_features"].size(1),-1)) # (bs, 128, em)
        oracle_action_tokens = self.action_encoder(observations["gt_actions"].long())

        space_tokens =  torch.cat((rgb_tokens, depth_tokens), dim=1) # naively concat

        # # noising oracle_action_tokens
        noise = torch.randn(oracle_action_tokens.shape, device=oracle_action_tokens.device)

        print(f"noising_timesteps min: {self.noise_scheduler.config.num_train_timesteps}")
        # noising_timesteps = torch.randint(
        #     0,
        #     self.noise_scheduler.config.num_train_timesteps,
        #     (len(noise),), device=noise.device
        # ).long()

        # print(f"alphas_cumprod shape: {noising_timesteps}")
       
        # assert 1==2

        # noised_orc_action_tokens = self.noise_scheduler.add_noise(
        #     oracle_action_tokens, noise,
        #     noising_timesteps
        # )

        assert 1==2

        # predict noise
        tokens = (instr_tokens,space_tokens)
        pred = self.predict_noise(tokens,noised_orc_action_tokens,noising_timesteps)






        context_features = self.cross_attention()

        # Cross-attention between instruction and visual features
        tokens = tokens.transpose(0, 1)  # Required for multihead attention


        tokens, _ = self.cross_attention(tokens, tokens, tokens)
        tokens, _ = self.self_attention(tokens, tokens, tokens)
        tokens = tokens.transpose(0, 1)  # Back to (bs, num_tokens, embedding_dim)

        if run_inference:
            return self.sample_trajectory(tokens)

        # Encode gt_actions
        encoded_gt_actions = self.action_encoder(gt_actions)  # (bs, 6, embedding_dim)

        # Add noise to gt_actions during training
        noise = torch.randint(0, self.num_actions, gt_actions.shape).to(gt_actions.device)  # Discrete noise
        timesteps = torch.randint(0, self.n_steps, (bs,)).long().to(gt_actions.device)

        noisy_actions = self.add_discrete_noise(encoded_gt_actions, noise, timesteps)

        # Predict the logits for the noisy actions
        predicted_logits = self.predict_noise(tokens, noisy_actions, timesteps)

        # Compute loss (cross entropy between predicted logits and original gt_actions)
        loss = F.cross_entropy(predicted_logits.view(-1, self.num_actions), gt_actions.view(-1))
        return loss

    def add_discrete_noise(self, actions, noise, timesteps):
        # Add noise to the actions; in this context, we are replacing actions with noisy actions
        # based on the timestep, simulating the forward process of diffusion.
        noisy_actions = actions + noise
        noisy_actions = noisy_actions % self.num_actions  # Ensure within valid action range
        return noisy_actions


    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats


    def predict_noise(self, tokens, noisy_actions, timesteps): # tokens in form (instr_tokens,space_tokens)

        
        context_features = self.vision_language_attention(tokens[1],tokens[0])

        
        time_embeddings = self.time_emb(timesteps.unsqueeze(-1).float())

        print(f" context features  {context_features.shape}")

        assert 1==2
        # Optionally add time embeddings
        time_embeddings = self.time_emb(timesteps.unsqueeze(-1).float())
        noisy_actions = noisy_actions + time_embeddings

        # Concatenate tokens with noisy_actions for the transformer
        transformer_input = torch.cat([tokens, noisy_actions.unsqueeze(1)], dim=1)

        # Use attention layers for predicting the logits
        transformer_input = transformer_input.transpose(0, 1)  # Required for multihead attention
        transformer_input, _ = self.self_attention(transformer_input, transformer_input, transformer_input)
        transformer_input = transformer_input.transpose(0, 1)  # Back to (bs, num_tokens, embedding_dim)
        
        logits = self.action_predictor(transformer_input[:, -1, :])

        return logits

    def sample_trajectory(self, tokens):
        bs = tokens.size(0)
        # Start from pure noise
        trajectory = torch.randint(0, self.num_actions, (bs, 1)).to(tokens.device)  # Start with random discrete actions

        # Iterative denoising process
        for t in reversed(range(self.n_steps)):
            timestep = torch.tensor([t], device=tokens.device).repeat(bs)
            time_embeddings = self.time_emb(timestep.unsqueeze(-1).float())
            trajectory = trajectory + time_embeddings

            transformer_input = torch.cat([tokens, trajectory.unsqueeze(1)], dim=1)

            # Use attention layers for denoising
            transformer_input = transformer_input.transpose(0, 1)  # Required for multihead attention
            transformer_input, _ = self.self_attention(transformer_input, transformer_input, transformer_input)
            transformer_input = transformer_input.transpose(0, 1)  # Back to (bs, num_tokens, embedding_dim)

            logits = self.action_predictor(transformer_input[:, -1, :])
            trajectory = torch.argmax(logits, dim=-1)  # Choose the most likely action

        return trajectory
