import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from habitat_baselines.common.baseline_registry import baseline_registry

from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule,ParallelAttention
from habitat_baselines.rl.ppo.policy import Policy
from diffuser_baselines.models.encoders.instruction_encoder import InstructionEncoder
from diffuser_baselines.models.common.position_encodings import RotaryPositionEncoding,SinusoidalPosEmb, PositionalEncoding


@baseline_registry.register_policy
class DiffusionPolicy(Policy):
    
    def __init__(
        self, config,
        num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps
    ) -> None:
        
        super(Policy, self).__init__()
        self.navigator = DiffusionNavigator(config,num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps)
        self.config = config

    def act(self,batch):

        # format batch data
        collected_data = {
        'instruction': [],
        'rgb_features': [],
        'depth_features': [],
        'gt_actions': []
        }

        self.navigator.encode_visions(batch,self.config)

        # for sample in batch:
        #     len_seq = sample[0]['instruction'].shape[0]
        
        # # randomly sample timestep t in the range [0, len_seq-1]
        # t = random.randint(0, len_seq - 1)
        
        # # Handle instruction, rgb_features, depth_features
        # collected_data['instruction'].append(torch.tensor(sample[0]['instruction'][t]))
        # collected_data['rgb_features'].append(torch.tensor(sample[0]['rgb_features'][t]))
        # collected_data['depth_features'].append(torch.tensor(sample[0]['depth_features'][t]))
        
        # # Handle gt_actions by selecting from t to t+F, padding with -1 if out of bounds
        # if t + F < len_seq:
        #     gt_action_segment = sample[2][t:t+F+1]
        # else:
        #     gt_action_segment = sample[2][t:] 
        #     padding_size = (t + F + 1) - len_seq 
        #     gt_action_segment = np.concatenate([gt_action_segment, np.full(padding_size, 0)]) # padding with STOP action if exceed

        # collected_data['gt_actions'].append(torch.tensor(gt_action_segment))
    
        # # Stack into batched tensors
        # collected_data['instruction'] = torch.stack(collected_data['instruction'], dim=0)
        # collected_data['rgb_features'] = torch.stack(collected_data['rgb_features'], dim=0)
        # collected_data['depth_features'] = torch.stack(collected_data['depth_features'], dim=0)
        # collected_data['gt_actions'] = torch.stack(collected_data['gt_actions'], dim=0)


        
    

    def build_loss(self,observations):

        loss = self.navigator(observations)

        return loss
    
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

        # prepare action embedding targets for inference
        action_targets = torch.arange(4).unsqueeze(0)
        self.action_em_targets = self.action_encoder(action_targets)

        # positional embeddings
        self.pe_layer = PositionalEncoding(embedding_dim,0)

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
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attention_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        self.cross_attention = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)
        self.self_attention = FFWRelativeSelfAttentionModule(embedding_dim,num_attention_heads,num_layers)

        # Diffusion schedulers
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
        )


        # predictors
        self.noise_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.n_steps = diffusion_timesteps
    

    def tokenlize_input(self,observations):

        bs = observations["instruction"].size(0)

        
        # tokenlize
        instr_tokens = self.instruction_encoder(observations["instruction"])  # (bs, embedding_dim)
        rgb_tokens = self.rgb_linear(observations["rgb_features"].view(bs,observations["rgb_features"].size(1),-1))  # (bs, 2048, em)
        depth_tokens = self.depth_linear(observations["depth_features"].view(bs,observations["depth_features"].size(1),-1)) # (bs, 128, em)

        if observations["gt_actions"] == None: # inference
            oracle_action_tokens = None
        else:
            oracle_action_tokens = self.action_encoder(observations["gt_actions"].long())

        return instr_tokens,rgb_tokens,depth_tokens,oracle_action_tokens




    def forward(self, observations, run_inference=False):

        
        # tokenlize
        instr_tokens,rgb_tokens,depth_tokens,oracle_action_tokens = self.tokenlize_input(observations)


        # inference _____
        
        if run_inference:
            return

        # train _____

        # noising oracle_action_tokens
        noise = torch.randn(oracle_action_tokens.shape, device=oracle_action_tokens.device)

        noising_timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

 
        noised_orc_action_tokens = self.noise_scheduler.add_noise(
            oracle_action_tokens, noise,
            noising_timesteps
        )



        # predict noise
        tokens = (instr_tokens,rgb_tokens,depth_tokens)
        pred = self.predict_noise(tokens,noised_orc_action_tokens,noising_timesteps)

        # compute loss
        loss = F.l1_loss(pred, noise, reduction='mean')

        return loss



    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats
    


    def predict_noise(self, tokens, noisy_actions, timesteps): # tokens in form (instr_tokens,rgb,depth)

        time_embeddings = self.time_emb(timesteps.float())
        
        # positional embedding
        instruction_position = self.pe_layer(tokens[0])
        action_position = self.pe_layer(noisy_actions)
        
        # observation features
        obs_features = self.cross_attention(query=tokens[1].transpose(0, 1),
            value=tokens[2].transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings)[-1].transpose(0,1) # takes last layer and return to (B,L,D)
        
        # context features 
        context_features = self.vision_language_attention(obs_features,instruction_position) # rgb attend instr.


        # action features
        action_features, _ = self.traj_lang_attention[0](
                seq1=action_position, seq1_key_padding_mask=None,
                seq2=instruction_position, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
        )


        # final features
        features = self.cross_attention(query=action_features.transpose(0, 1),
            value=context_features.transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings)[-1].transpose(0,1)
        
        
        final_features = self.self_attention(features.transpose(0,1), diff_ts=time_embeddings,
                query_pos=None, context=None, context_pos=None)[-1].transpose(0,1)


        noise_prediction = self.noise_predictor(final_features)

        return noise_prediction


    def retrive_action_from_em(self,embeddings):  # retrive action index from embedding
        distances = torch.cdist(self.action_em_targets, embeddings)
        actions_indexs = torch.argmin(distances, dim=-1)
        return actions_indexs

    def inference_actions(self,pred_noises): # pred_noises (B,N,D)

        self.noise_scheduler.set_timesteps(self.n_steps)

        pure_noise = torch.randn(
            size=pred_noises.shape,
            dtype=pred_noises.dtype,
            device=pred_noises.device
        )

        intermidiate_noise = pure_noise

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            denoised_action_em = self.noise_scheduler.step(
                pred_noises, t, intermidiate_noise
            ).prev_sample
            intermidiate_noise = denoised_action_em

        # return action index
        actions = self.retrive_action_from_em(intermidiate_noise)
        return actions


    def encode_visions(self,batch,config):


        # init frozon encoders
        # init the depth visual encoder
        from diffuser_baselines.models.encoders import resnet_encoders
        from gym import spaces

        obs_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype='uint8'),
            "depth": spaces.Box(low=0, high=255, shape=(256, 256, 1), dtype='uint8')
        })

        assert config.MODEL.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]
        self.depth_encoder = getattr(
            resnet_encoders, config.MODEL.DEPTH_ENCODER.cnn_type
        )(
            obs_space,
            output_size=config.MODEL.DEPTH_ENCODER.output_size,
            checkpoint=config.MODEL.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=config.MODEL.DEPTH_ENCODER.backbone,
            trainable=config.MODEL.DEPTH_ENCODER.trainable,
            spatial_output=True,
        )

        # init the RGB visual encoder
        assert config.MODEL.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ]
        self.rgb_encoder = getattr(
            resnet_encoders, config.MODEL.RGB_ENCODER.cnn_type
        )(
            config.MODEL.RGB_ENCODER.output_size,
            normalize_visual_inputs=config.MODEL.normalize_rgb,
            trainable=config.MODEL.RGB_ENCODER.trainable,
            spatial_output=True,
        )

        self.depth_encoder.to(self.device)
        self.rgb_encoder.to(self.device)

        depth_embedding = self.depth_encoder(batch)

        print(f"depth features {depth_embedding.shape}")

        return None