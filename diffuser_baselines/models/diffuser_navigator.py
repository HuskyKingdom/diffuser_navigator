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

    def act(self,batch,t=None):

        

        rgb_features,depth_features = self.navigator.encode_visions(batch,self.config)

        # format batch data
        collected_data = {
        'instruction': batch['instruction'],
        'rgb_features': rgb_features.to(batch['instruction'].device),
        'depth_features': depth_features.to(batch['instruction'].device),
        'gt_actions': None,
        'seq_timesteps': torch.tensor([t]).to(batch['instruction'].device),
        }


        actions = self.navigator(collected_data,run_inference = True)

        print(f"final actions {actions}")
        return actions

    


        
    

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

        self.config = config
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions

        self.total_evaled = 0
        self.total_correct = 0

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

        self.seq_leng_emb = nn.Sequential(
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
        self.pe_layer = PositionalEncoding(embedding_dim,0.2)

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

        self.language_self_atten = FFWRelativeSelfAttentionModule(embedding_dim,num_attention_heads,num_layers)
        self.cross_attention = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)
        self.cross_attention_sec = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)
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

        seq_leng_features = self.seq_leng_emb(observations["seq_timesteps"])


        if observations["gt_actions"] == None: # inference
            oracle_action_tokens = None
        else:
            oracle_action_tokens = self.action_encoder(observations["gt_actions"].long())

        

        return instr_tokens,rgb_tokens,depth_tokens,oracle_action_tokens,seq_leng_features




    def forward(self, observations, run_inference=False):

        
        # tokenlize
        instr_tokens,rgb_tokens,depth_tokens,oracle_action_tokens,seq_leng_features = self.tokenlize_input(observations)

        # language padding mask
        pad_mask = (observations['instruction'] == 0)


        # inference _____
        
        if run_inference:
            tokens = (instr_tokens,rgb_tokens,depth_tokens,seq_leng_features)
            return self.inference_actions(tokens,pad_mask)

        # train _____

        # noising oracle_action_tokens
        noise = torch.randn(oracle_action_tokens.shape, device=oracle_action_tokens.device)

        noising_timesteps = torch.randint(
            600,
            700, # self.noise_scheduler.config.num_train_timesteps
            (len(noise),), device=noise.device
        ).long()

 
        noised_orc_action_tokens = self.noise_scheduler.add_noise(
            oracle_action_tokens, noise,
            noising_timesteps
        )



        # predict noise
        tokens = (instr_tokens,rgb_tokens,depth_tokens,seq_leng_features)
        pred = self.predict_noise(tokens,noised_orc_action_tokens,noising_timesteps,pad_mask)


        # evaluations ____


        # noised_orc_action_tokens = torch.randn(
        #     size=(len(tokens[0]),self.config.DIFFUSER.action_length,self.config.DIFFUSER.embedding_dim), # (bs, L, emb.)
        #     dtype=tokens[0].dtype,
        #     device=tokens[0].device
        # )


        print(f"GroundTruth Actions {observations['gt_actions'][0]}")
        

        denoise_steps = list(range(noising_timesteps[0].item(), -1, -1))

        tokens = (instr_tokens[0].unsqueeze(0),rgb_tokens[0].unsqueeze(0),depth_tokens[0].unsqueeze(0),seq_leng_features[0].unsqueeze(0))
        intermidiate_noise = noised_orc_action_tokens[0].unsqueeze(0)


        pad_mask = pad_mask[0].unsqueeze(0)
    
        for t in denoise_steps:

            # noise pred.
            with torch.no_grad():
                pred_noises = self.predict_noise(tokens,intermidiate_noise,t * torch.ones(len(tokens[0])).to(tokens[0].device).long(),pad_mask)

            step_out = self.noise_scheduler.step(
                pred_noises, t, intermidiate_noise
            )

            intermidiate_noise = step_out["prev_sample"]

  
        denoised = step_out["prev_sample"]
        pre_actions = self.retrive_action_from_em(denoised)
        print(f"Predicted Actions {pre_actions}")


        # analyzing
        list1 = pre_actions.squeeze(0).cpu().tolist()
        list2 = observations['gt_actions'][0].cpu().tolist()

        same_index_count = sum(1 for a, b in zip(list1, list2) if a == b)
        self.total_correct += same_index_count

        if self.total_evaled < 100:
            self.total_evaled += 3
        else:
            print(self.total_correct)
            print(f"evaluated {self.total_evaled} | accuracy {self.total_correct / (self.total_evaled)}")
            assert 1==2


        # compute loss
        # kl_loss = F.kl_div(pred.log_softmax(dim=-1), noise.softmax(dim=-1), reduction='batchmean')
        mse_loss = F.mse_loss(pred, noise)

        # loss = mse_loss + self.config.DIFFUSER.beta * kl_loss
        loss = mse_loss

        loss = loss - loss # evaluation

        return loss



    def vision_language_attention(self, feats, instr_feats,seq1_pad=None,seq2_pad=None):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=seq1_pad,
            seq2=instr_feats, seq2_key_padding_mask=seq2_pad,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats
    


    def predict_noise(self, tokens, noisy_actions, timesteps,pad_mask): # tokens in form (instr_tokens,rgb,depth,seq_leng_features)

        time_embeddings = self.time_emb(timesteps.float())
        
        # positional embedding
        instruction_position = self.pe_layer(tokens[0])
        action_position = self.pe_layer(noisy_actions)

        # languege features
        lan_features = self.language_self_atten(instruction_position.transpose(0,1), diff_ts=tokens[-1],
                query_pos=None, context=None, context_pos=None,pad_mask=pad_mask)[-1].transpose(0,1)
        
        # observation features
        obs_features = self.cross_attention(query=tokens[1].transpose(0, 1),
            value=tokens[2].transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings)[-1].transpose(0,1) # takes last layer and return to (B,L,D)
        
        # context features 
        context_features = self.vision_language_attention(obs_features,lan_features,seq2_pad=pad_mask) # rgb attend instr.


        # action features
        action_features, _ = self.traj_lang_attention[0](
                seq1=action_position, seq1_key_padding_mask=None,
                seq2=lan_features, seq2_key_padding_mask=pad_mask,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
        )


        # final features
        features = self.cross_attention_sec(query=action_features.transpose(0, 1),
            value=context_features.transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings)[-1].transpose(0,1)
        
        
        final_features = self.self_attention(features.transpose(0,1), diff_ts=time_embeddings,
                query_pos=None, context=None, context_pos=None)[-1].transpose(0,1)


        noise_prediction = self.noise_predictor(final_features)

        return noise_prediction


    # def retrive_action_from_em(self,embeddings):  # retrive action index from embedding
        
    #     target = self.action_em_targets.to(embeddings.device)
    #     l1_dist = torch.abs(target.unsqueeze(1) - embeddings.unsqueeze(2)).sum(dim=-1)  # (B, L, A)

    #     actions_indexs = torch.argmin(l1_dist, dim=-1)

    #     return actions_indexs


    def retrive_action_from_em(self, embeddings):  # retrieve action index from embedding
        target = self.action_em_targets.to(embeddings.device)
        
        # Compute L2 distance (squared differences) instead of L1
        l2_dist = torch.sqrt(torch.sum((target.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1))  # (B, L, A)
        
        actions_indexs = torch.argmin(l2_dist, dim=-1)

        return actions_indexs


    # # consine sim
    # def retrive_action_from_em(self, embeddings):  # retrieve action index from embedding
    #     target = self.action_em_targets.to(embeddings.device)
        
    #     # Normalize the embeddings and targets
    #     embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=-1)  # (B, L, D)
    #     target_norm = torch.nn.functional.normalize(target, p=2, dim=-1)  # (A, D)
        
    #     # Compute cosine similarity
    #     cos_sim = torch.matmul(embeddings_norm, target_norm.transpose(-1, -2))  # (B, L, A)
        
    #     # Since we want to retrieve the action index, use argmax to find the most similar action
    #     actions_indexs = torch.argmax(cos_sim, dim=-1)
        
    #     return actions_indexs



    def inference_actions(self,tokens,mask): # pred_noises (B,N,D)

        self.noise_scheduler.set_timesteps(self.n_steps)

        

        pure_noise = torch.randn(
            size=(len(tokens[0]),self.config.DIFFUSER.action_length,self.config.DIFFUSER.embedding_dim), # (bs, L, emb.)
            dtype=tokens[0].dtype,
            device=tokens[0].device
        )

        intermidiate_noise = pure_noise

        # Iterative denoising
        timesteps = self.noise_scheduler.timesteps

        for t in timesteps:
            
            # noise pred.
            with torch.no_grad():
                pred_noises = self.predict_noise(tokens,intermidiate_noise,t * torch.ones(len(tokens[0])).to(tokens[0].device).long(),mask)

            
            step_out = self.noise_scheduler.step(
                pred_noises, t, intermidiate_noise
            )

            intermidiate_noise = step_out["prev_sample"]



        # return action index
        actions = self.retrive_action_from_em(intermidiate_noise)

        return actions


    def encode_visions(self,batch,config):

        # TODO MOVE TO INIT.
        
        # init frozon encoders
        # init the depth visual encoder
        from diffuser_baselines.models.encoders import resnet_encoders
        from gym import spaces

        obs_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype='float32'),
            "depth": spaces.Box(low=0, high=255, shape=(256, 256, 1), dtype='float32')
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

        self.depth_encoder.to(next(self.parameters()).device).eval()
        self.rgb_encoder.to(next(self.parameters()).device).eval()


        # hooking cnn output
        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())

            return hook

        rgb_features = None
        rgb_hook = None    
        rgb_features = torch.zeros((1,), device="cpu")
        rgb_hook = self.rgb_encoder.cnn.register_forward_hook(
            hook_builder(rgb_features)
        )

        depth_features = None
        depth_hook = None
        depth_features = torch.zeros((1,), device="cpu")
        depth_hook = self.depth_encoder.visual_encoder.register_forward_hook(
            hook_builder(depth_features)
        )

        depth_embedding = self.depth_encoder(batch)
        rgb_embedding = self.rgb_encoder(batch)


        rgb_hook.remove()
        depth_hook.remove()


        return rgb_features,depth_features