import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from habitat_baselines.common.baseline_registry import baseline_registry

from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule,ParallelAttention
from habitat_baselines.rl.ppo.policy import Policy
from diffuser_baselines.models.encoders.instruction_encoder import InstructionEncoder
from diffuser_baselines.models.common.position_encodings import RotaryPositionEncoding,SinusoidalPosEmb, PositionalEncoding

from diffuser_baselines.models.encoders import resnet_encoders
from diffuser_baselines.models.encoders.his_encoder import HistoryGRU

from gym import spaces
import numpy as np

@baseline_registry.register_policy
class DiffusionPolicy(Policy):
    
    def __init__(
        self, config,
        num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps
    ) -> None:
        
        super(Policy, self).__init__()
        self.navigator = DiffusionNavigator(config,num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps)
        self.config = config

    def act(self,batch, all_pose=None, hiddens = None, encode_only=False,print_info=False):

        
        rgb_features,depth_features = self.navigator.encode_visions(batch,self.config) # raw batch

        if encode_only:
            return None

        # format batch data
        collected_data = {
        'instruction': batch['instruction'],
        'rgb_features': rgb_features.to(batch['instruction'].device),
        'depth_features': depth_features.to(batch['instruction'].device),
        'proprioceptions': torch.tensor(all_pose,dtype=torch.float32).to(batch['instruction'].device)
        }


        actions, next_hidden = self.navigator(collected_data,run_inference = True,hiddens=hiddens,print_info=print_info)

        return actions, next_hidden
        
    

    def build_loss(self,observations):

        rgb_features,depth_features = self.navigator.encode_visions(observations,self.config) # stored vision features


        # format batch data
        collected_data = {
        'instruction': observations['instruction'],
        'rgb_features': rgb_features.to(observations['instruction'].device),
        'depth_features': depth_features.to(observations['instruction'].device),
        'gt_actions': observations['gt_actions'],
        'histories': observations['histories'].to(observations['instruction'].device),     # (bs,max_seq_len,32768)      
        'his_len': observations['his_len'].to(observations['instruction'].device),
        'trajectories': observations['trajectories'].to(observations['instruction'].device),
        'proprioceptions': observations['proprioceptions'].to(observations['instruction'].device)
        }

        loss = self.navigator(collected_data)

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


        # Vision Encoders _____________________
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

        self.depth_encoder.to(next(self.parameters()).device).train()
        self.rgb_encoder.to(next(self.parameters()).device).train()



        # Other Encoders
        self.instruction_encoder = InstructionEncoder(config,embedding_dim)
        self.rgb_linear = nn.Linear(2112,embedding_dim)
        self.depth_linear = nn.Linear(192,embedding_dim)

        # self.action_encoder = nn.Embedding(num_actions, embedding_dim)
        self.traj_encoder = nn.Sequential(
            nn.Linear(self.config.DIFFUSER.traj_space, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(self.config.DIFFUSER.traj_space, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )


        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.his_encoder = HistoryGRU(32768,512,embedding_dim,2)

        # for param in self.action_encoder.parameters(): # freeze action representations
        #     param.requires_grad = False

        # prepare action embedding targets for inference
        # action_targets = torch.arange(self.config.DIFFUSER.action_space).unsqueeze(0)
        # self.action_em_targets = self.action_encoder(action_targets)

        # positional embeddings
        self.pe_layer = PositionalEncoding(embedding_dim,0.2)

        # Attention layers _____________________
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
        self.obs_cross_attention = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)
        self.context_cross_attention = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)
        # self.self_attention = FFWRelativeSelfAttentionModule(embedding_dim,num_attention_heads,num_layers)
        self.term_self_atten = FFWRelativeSelfAttentionModule(embedding_dim,num_attention_heads,num_layers)
        self.noise_self_atten = FFWRelativeSelfAttentionModule(embedding_dim,num_attention_heads,num_layers)

        # Diffusion schedulers
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
        )


        # predictors (history emb + current emb)
        self.term_projector = nn.Linear(embedding_dim*2, embedding_dim)
        self.termination_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), # (bs,3,64)
            nn.ReLU(),
            nn.Linear(embedding_dim, 1), # (bs,3,1)
            nn.Sigmoid()
        )

        self.noise_projector = nn.Linear(embedding_dim*2, embedding_dim)
        self.noise_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, self.config.DIFFUSER.traj_space)
        )

        self.n_steps = diffusion_timesteps
    


    def normalize_dim(self, tensor, min_val=None, max_val=None, feature_range=(-1, 1)):
    
        # this function and denorm function would automatically extend the input tensor to shape [bs,len,4]

        if min_val is None:
            min_val = torch.tensor([[[-32.31, -5.96, -74.19, -3.15]]], dtype=torch.float32,device = tensor.device)
        if max_val is None:
            max_val = torch.tensor([[[70.04, 7.46, 46.58, 3.15]]], dtype=torch.float32,device = tensor.device) 
        
        # norm to [0, 1]
        norm_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
        
        # map to feature range
        scale = feature_range[1] - feature_range[0]
        norm_tensor = norm_tensor * scale + feature_range[0]
        
        return norm_tensor


    def denormalize_dim(self, tensor, min_val=None, max_val=None, feature_range=(-1, 1)):

        if min_val is None:
            min_val = torch.tensor([[[-32.31, -5.96, -74.19, -3.15]]], dtype=torch.float32,device = tensor.device)
        if max_val is None:
            max_val = torch.tensor([[[70.04, 7.46, 46.58, 3.15]]], dtype=torch.float32,device = tensor.device) 


        scale = feature_range[1] - feature_range[0]
        tensor = (tensor - feature_range[0]) / scale

        return tensor * (max_val - min_val) + min_val


    def encode_trajectories(self,traj):
        traj_tokens = self.traj_encoder(traj)
        return traj_tokens


    def tokenlize_input(self,observations,hiddens,inference=False):

        bs = observations["instruction"].size(0)
        
        # tokenlize
        instr_tokens = self.instruction_encoder(observations["instruction"])  # (bs, embedding_dim)

        rgb_features =  observations["rgb_features"].view(bs,observations["rgb_features"].size(1),-1).permute(0,2,1)
        depth_features =  observations["depth_features"].view(bs,observations["depth_features"].size(1),-1).permute(0,2,1)


        rgb_tokens = self.rgb_linear(rgb_features)  # (bs, 16, 2112) -> (bs, 16, em)
        depth_tokens = self.depth_linear(depth_features) # (bs, 16, 192) -> (bs, 16, em)


        traj_tokens = None # will be encoded later
        pose_feature = self.pose_encoder(observations["proprioceptions"]) 


        if inference: # dont pack
            input_x = observations["rgb_features"][:, :2048, :, :].view(observations["rgb_features"].shape[0],-1).unsqueeze(1) # take current as input and reshape to (bs,len,d)
            history_tokens, next_hiddens = self.his_encoder(input_x,hiddens,inference=True)
        else:
            history_tokens, next_hiddens = self.his_encoder(observations["histories"],hiddens,observations["his_len"],inference = False)


   
        tokens = [instr_tokens,rgb_tokens,depth_tokens,history_tokens,traj_tokens,pose_feature]

        return tokens, next_hiddens


    def forward(self, observations, run_inference=False,hiddens=None, print_info = False):


        observations['proprioceptions'] = self.normalize_dim(observations['proprioceptions']).squeeze(0) # normalize input alone dimensions


        # language padding mask
        pad_mask = (observations['instruction'] == 0)

        # inference _____
        if run_inference:
            return self.inference_actions(observations,pad_mask,hiddens,print_info)

        # train _____
        observations["trajectories"] = self.normalize_dim(observations["trajectories"]) # normalize input alone dimensions

        # buiding noise
        noise = torch.randn(observations["trajectories"].shape, device=observations["trajectories"].device)

        noising_timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps, # self.noise_scheduler.config.num_train_timesteps
            (len(noise),), device=noise.device
        ).long()

        noised_traj = self.noise_scheduler.add_noise(
            observations["trajectories"], noise,
            noising_timesteps
        )

        # tokenlize
        tokens, _ = self.tokenlize_input(observations,hiddens)
        # encode traj
        tokens[4] = self.encode_trajectories(noised_traj)

        # predict noise
        pred_noise, pred_termination = self.predict_noise(tokens,noising_timesteps,pad_mask)


        
        target_terminations = torch.where(observations["gt_actions"] == 0, torch.tensor(1, device=observations["gt_actions"].device), torch.tensor(0, device=observations["gt_actions"].device)).float()
        
        # compute loss
        mse_loss = F.mse_loss(pred_noise, noise)
        bin_crossentro_loss = F.binary_cross_entropy(pred_termination, target_terminations)

        # loss = mse_loss + self.config.DIFFUSER.beta * kl_loss
        loss = mse_loss + bin_crossentro_loss


        # # # evaluations ____

        # loss = loss - loss # ignore update 
        # noise = torch.randn(observations["trajectories"][0].unsqueeze(0).shape, device=observations["trajectories"].device)

        # noising_timesteps = torch.randint(
        #     99,
        #     100, # self.noise_scheduler.config.num_train_timesteps
        #     (1,), device=noise.device
        # ).long()

        # noised_traj = self.noise_scheduler.add_noise(
        #     observations["trajectories"][0].unsqueeze(0), noise,
        #     noising_timesteps
        # )
        

        # denoise_steps = list(range(noising_timesteps[0].item(), -1, -1))
        # intermidiate_noise = noise

        # with torch.no_grad():
        #     tokens, _ = self.tokenlize_input(observations,hiddens) # dont pack
        
        

        # for t in denoise_steps:
        #     # noise pred.
        #     with torch.no_grad():

        #         # encode traj and predict noise
        #         tokens[4] = self.encode_trajectories(intermidiate_noise) # dont pack

        #         tokens = [tokens[0][0].unsqueeze(0),tokens[1][0].unsqueeze(0),tokens[2][0].unsqueeze(0),tokens[3][0].unsqueeze(0),tokens[4][0].unsqueeze(0),tokens[5][0].unsqueeze(0)]
        #         pad_mask = pad_mask[0].unsqueeze(0)

        #         pred_noises,pred_termination = self.predict_noise(tokens,t * torch.ones(len(tokens[0])).to(tokens[0].device).long(),pad_mask)

        #     step_out = self.noise_scheduler.step(
        #         pred_noises, t, intermidiate_noise
        #     )


        #     intermidiate_noise = step_out["prev_sample"]

        # denormed_groundtruth = self.denormalize_dim(observations['trajectories'][0])
        # denormed_pred = self.denormalize_dim(intermidiate_noise)

        # print(f"GroundTruth Trajectory {denormed_groundtruth} | Predicted Actions {denormed_pred}")
        # print(f"ground truth actions {target_terminations[0]} | predicted terminations {pred_termination}")


        # # analyzing
        # if self.total_evaled < 200:
        #     self.total_evaled += 3
        # else:
        #     assert 1==2



        return loss



    def vision_language_attention(self, feats, instr_feats,seq1_pad=None,seq2_pad=None):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=seq1_pad,
            seq2=instr_feats, seq2_key_padding_mask=seq2_pad,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats
    

    def predict_noise(self, tokens, timesteps,pad_mask): # tokens in form (instr_tokens,rgb,depth,history_tokens,traj,pose_feature)

        time_embeddings = self.time_emb(timesteps.float())
        time_embeddings = time_embeddings + tokens[-1] # fused (48,64)

        # positional embedding
        instruction_position = self.pe_layer(tokens[0])
        traj_position = self.pe_layer(tokens[4])


        # languege features
        lan_features = self.language_self_atten(instruction_position.transpose(0,1), diff_ts=time_embeddings,
                query_pos=None, context=None, context_pos=None,pad_mask=pad_mask)[-1].transpose(0,1)
        

        # observation features
        obs_features = self.obs_cross_attention(query=tokens[1].transpose(0, 1),
            value=tokens[2].transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings)[-1].transpose(0,1) # takes last layer and return to (B,L,D)
        
        # context features 
        context_features = self.vision_language_attention(obs_features,lan_features,seq2_pad=pad_mask) # rgb attend instr.


        # trajectory features
        traj_features, _ = self.traj_lang_attention[0](
                seq1=traj_position, seq1_key_padding_mask=None,
                seq2=lan_features, seq2_key_padding_mask=pad_mask,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
        )


        # final features
        features = self.context_cross_attention(query=traj_features.transpose(0, 1),
            value=context_features.transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings)[-1].transpose(0,1) # (bs,3,64)
        

        # fuse with history
        history_feature = tokens[-3].unsqueeze(1).expand(-1,features.shape[1],-1)
        fused_feature = torch.cat((features,history_feature),dim=-1) # (bs,seq_len,d*2)

        # predicting terminationx
        termination_feature = self.term_projector(fused_feature)
        termination_feature  = self.term_self_atten(termination_feature.transpose(0,1), diff_ts=time_embeddings,
                                                                    query_pos=None, context=None, context_pos=None)[-1].transpose(0,1)
        termination_prediction = self.termination_predictor(termination_feature).view(termination_feature.shape[0],-1) # (bs,seq_len,1) -> (bs,seq_len)

                
        # final_features = self.self_attention(fused_feature.transpose(0,1), diff_ts=time_embeddings,
        #         query_pos=None, context=None, context_pos=None)[-1].transpose(0,1)
        
        # print(final_features.shape)
        # assert 1==2

        
        
        noise_feature = self.noise_projector(fused_feature)
        noise_feature  = self.noise_self_atten(noise_feature.transpose(0,1), diff_ts=time_embeddings,
                                                                    query_pos=None, context=None, context_pos=None)[-1].transpose(0,1)
        noise_prediction = self.noise_predictor(noise_feature) # (bs,seq_len,traj_space)

        return noise_prediction,termination_prediction


    def calculate_actions(self, tensor, head_threshold):

        bs, seq_len, dim = tensor.shape

        actions = torch.zeros((bs, seq_len - 1), dtype=torch.int)

        for i in range(bs):
            for j in range(seq_len - 1):

                current_heading = tensor[i, j, 3] 
                next_heading = tensor[i, j + 1, 3] 

                diff_heading = next_heading - current_heading
  
                if diff_heading >= head_threshold:
                    actions[i, j] = 2
                elif diff_heading <= -head_threshold:
                    actions[i, j] = 3
                else:
                    actions[i, j] = 1

        return actions


    def traj_to_action(self,pose,traj,terminations):

        # param. pose in shape (B,4); traj in shape (B,L,4)
        # return. actions in shape (B,L)

        head_threshold = 0.2618


        full_traj = torch.cat((pose,traj),1)

        actions = self.calculate_actions(full_traj, head_threshold)

        high_prob_mask = terminations >= 0.1
        sampled_mask = torch.ones_like(terminations)  # 初始化全 1 的 mask


        sampled_mask[high_prob_mask] = torch.bernoulli(1 - terminations[high_prob_mask]) # prob. of not terminates

        # sampled_mask = torch.bernoulli(1 - terminations)  # prob. of not terminates

        actions[sampled_mask == 0] = 0 

        
        return actions


    def inference_actions(self,observations,mask,hiddens,print_info): # pred_noises (B,N,D)

        self.noise_scheduler.set_timesteps(self.n_steps)

        pure_noise = torch.randn(
            size=(len(observations['proprioceptions']),self.config.DIFFUSER.traj_length,self.config.DIFFUSER.traj_space), # (bs, L, 4)
            dtype=observations['proprioceptions'].dtype,
            device=observations['proprioceptions'].device
        )

        intermidiate_noise = pure_noise

        # Iterative denoising
        timesteps = self.noise_scheduler.timesteps

        # tokenlize input
        with torch.no_grad():
            tokens, next_hiddens = self.tokenlize_input(observations,hiddens,True) # dont pack

        for t in timesteps:
            
            # noise pred.
            with torch.no_grad():
                # encode traj and predict noise
                tokens[4] = self.encode_trajectories(intermidiate_noise) # dont pack
                pred_noises,pred_terminations = self.predict_noise(tokens,t * torch.ones(len(tokens[0])).to(tokens[0].device).long(),mask)

            step_out = self.noise_scheduler.step(
                pred_noises, t, intermidiate_noise
            )

            intermidiate_noise = step_out["prev_sample"]


        denoised = intermidiate_noise

        # return action index
        denomed_pose = self.denormalize_dim(observations['proprioceptions'])
        denormed_denoised = self.denormalize_dim(denoised)
        actions = self.traj_to_action(denomed_pose, denormed_denoised,pred_terminations)

        if print_info:
            print(f"final actions {actions} | t {pred_terminations}")

        return actions,next_hiddens


    def encode_visions(self,batch,config):

        depth_embedding = self.depth_encoder(batch)
        rgb_embedding = self.rgb_encoder(batch)

        return rgb_embedding,depth_embedding