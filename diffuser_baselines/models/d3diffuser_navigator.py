import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy
from diffuser_baselines.models.encoders.instruction_encoder import InstructionEncoder
from diffuser_baselines.models.encoders.trajectory_decoder import TrajectoryDecoder

from diffuser_baselines.models.encoders import resnet_encoders
from diffuser_baselines.models.utils import UnMaskedSoftmaxCELoss,MaskedWeightedLoss

from transformers import BertTokenizer      

from gym import spaces
import numpy as np

@baseline_registry.register_policy
class D3DiffusionPolicy(Policy):
    
    def __init__(
        self, config,
        num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps
    ) -> None:
        
        super(Policy, self).__init__()
        self.config = config
        self.navigator = D3DiffusionNavigator(config,num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps)
        self.rgb_his = []
        self.depth_his = []
        self.pre_actions = None
        

    def act(self,observations, prev_actions, encode_only=False,print_info = False,ins_text=None): 


        rgb_features,depth_features = self.navigator.encode_visions(observations,self.config) # raw batch

        # storing histories
        self.rgb_his.append(rgb_features)
        self.depth_his.append(depth_features)
        if self.pre_actions == None:
            self.pre_actions = prev_actions
        else:
            self.pre_actions = torch.cat((self.pre_actions,prev_actions),dim=1)

    
       
        if encode_only:
            return None
        
        

        # action inference

        rgb_features = torch.stack(self.rgb_his, dim=1)
        depth_features = torch.stack(self.depth_his, dim=1)


        # (B,T,C,H,W) -> (B+T,C,H,W)
        B,T,C,H,W = rgb_features.shape
        rgb_features = rgb_features.view(-1,C,H,W)
        B,T,C,H,W = depth_features.shape
        depth_features = depth_features.view(-1,C,H,W)
        
       
        # format batch data
        collected_data = {
        'instruction': observations['instruction'],
        'rgb_features': rgb_features.to(observations['instruction'].device),
        'depth_features': depth_features.to(observations['instruction'].device),
        'prev_actions': self.pre_actions.to(observations['instruction'].device).long(),
        'trajectories': None,
        'padding_mask': None,
        'lengths': None,
        'weights': None,
        'ins_text': ins_text
        }

        

        action = self.navigator(collected_data,(B,T), inference = True, ins_text=ins_text)
        

        if print_info:
            print(f"Action inferenced : {action.item()}, with history length {len(self.rgb_his)}")


        return action
        
    

    def build_loss(self,observations):


        # (B,T,C,H,W) -> (B+T,C,H,W)
        B,T,C,H,W = observations['rgb_features'].shape
        observations['rgb_features'] = observations['rgb_features'].view(-1,C,H,W)
        B,T,C,H,W = observations['depth_features'].shape
        observations['depth_features'] = observations['depth_features'].view(-1,C,H,W)
        
        


        rgb_features,depth_features = self.navigator.encode_visions(observations,self.config) # stored vision features
        
        

        # format batch data
        collected_data = {
        'instruction': observations['instruction'],
        'rgb_features': rgb_features.to(observations['instruction'].device),
        'depth_features': depth_features.to(observations['instruction'].device),
        'gt_actions': observations['gt_actions'],
        'prev_actions': observations['prev_actions'].to(observations['instruction'].device),
        'trajectories': observations['trajectories'].to(observations['instruction'].device),
        'padding_mask': observations['padding_mask'].to(observations['instruction'].device).bool(),
        'lengths': observations['lengths'].to(observations['instruction'].device),
        'weights': observations['weights'].to(observations['instruction'].device),
        'ins_text': observations['ins_text'],
        'progress': observations['progress'].to(observations['instruction'].device),
        }

        
        loss,actions_pred = self.navigator(collected_data,(B,T), inference = False)


        return loss,actions_pred
    
    @classmethod
    def from_config(
        cls,config, num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps
    ):
        return cls(
            config,num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps
        )
    
    def to_onehot(self,inputs):
        """
         (bs, seq_len) -> (bs, seq_len, num_cls) one-hot encoding
        """
        num_cls = self.config.DIFFUSER.action_space
        bs, seq_len = inputs.shape
        one_hot = torch.zeros(bs, seq_len, num_cls, device=inputs.device)  
        one_hot.scatter_(2, inputs.unsqueeze(-1), 1) 
        return one_hot
    
    def clear_his(self):

        self.rgb_his = []
        self.depth_his = []
        self.pre_actions = None




class D3DiffusionNavigator(nn.Module):


    def __init__(self, config,num_actions,embedding_dim,num_attention_heads,num_layers,diffusion_timesteps):

        super(D3DiffusionNavigator, self).__init__()

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


        self.rgb_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(33792, embedding_dim),
                nn.ReLU(),
            )
        
        self.depth_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3072, embedding_dim),
                nn.ReLU(),
            )

        decoder_dim = int(embedding_dim*2 + embedding_dim/2)

        # Encoders
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.instruction_encoder = InstructionEncoder(config,embedding_dim)
        self.action_encoder = nn.Embedding(num_actions + 1, int(embedding_dim/2)) # additional action as start token
        self.encoder_linear = nn.Linear(768,decoder_dim)
        
        # Decoder
        self.decoder = TrajectoryDecoder(config,decoder_dim,num_attention_heads,num_layers,num_actions)
        self.softmax_CE = UnMaskedSoftmaxCELoss()

        


        self.train()
        self.rgb_encoder.cnn.eval()
        self.depth_encoder.visual_encoder.eval()

    



    def get_context_feature(self,observations,inference=False):

        

        bs = observations["instruction"].size(0)

        rgb_features = self.rgb_linear(observations["rgb_features"])  # (B+T, channel, 4,4) -> (B+T, emb)
        depth_features = self.depth_linear(observations["depth_features"]) # (B+T, channel, 1,1) -> (B+T, emb)

        if not inference: # compute action featrues based on gt actions

            # construct input as [<start>,...]
            # action_except = observations['gt_actions'][:, :-1] # remove last element <end>
            
            action_except = observations['prev_actions'][:, 1:] # remove first
            action_start_token = torch.full((observations['prev_actions'].shape[0], 1), 4).to(observations['prev_actions'].device).long() # add start token
            action_input = torch.cat([action_start_token, action_except.long()], dim=1) # construct input
            action_input = action_input.view(-1,) # # (B,T) -> (B+T,)
            action_features = self.action_encoder(action_input.long()) # (B+T,) -> (B+T, emb)

        else: # compute action featrues based on decoder outputs (prev actions)
            
            action_except = observations['prev_actions'][:, 1:] # remove first
            action_start_token = torch.full((observations['prev_actions'].shape[0], 1), 4).to(observations['prev_actions'].device).long() # add start token
            action_input = torch.cat([action_start_token, action_except.long()], dim=1) # construct input
            action_input = action_input.view(-1,) # # (B,T) -> (B+T,)
            action_features = self.action_encoder(action_input.long()) # (B+T,) -> (B+T, emb)


        
        # print(rgb_features.shape,depth_features.shape,action_features.shape)
        observation_context = torch.cat((rgb_features,depth_features,action_features),dim=-1) # (B+T, emb*2+emb/2)
        
        return observation_context


    
    def generate_causal_mask(self, seq_length, device=None, dtype=None):
        mask = torch.triu(torch.full((seq_length, seq_length), float('-inf'), device=device, dtype=dtype), diagonal=1)
        return mask
                

    def forward(self, observations, dims, inference=False, ins_text=None):
        
        B,T = dims
        
        # tokenlize text
        batch_tokens = self.tokenizer(
            observations["ins_text"],
            padding="max_length",          
            truncation=True,       
            max_length=200,        
            return_tensors="pt"    
        )
        batch_tokens = {
                k: v.to(
                        device=observations["rgb_features"].device,
                        dtype=torch.int64,
                        non_blocking=True,
                    )
                for k, v in batch_tokens.items()
            }
        encoder_pad_mask = batch_tokens["attention_mask"] == 0 # to boolean mask


        if inference:

            # encoder
            # encoder_pad_mask = (observations['instruction'] == 0)
            enc_out = self.instruction_encoder(batch_tokens) # (bs,200,emd) | ins_text for visulization
            enc_out = self.encoder_linear(enc_out)

            # decoder
            context_feature = self.get_context_feature(observations,inference=inference)
            context_feature = context_feature.view(B,T,-1)
            causal_mask = self.generate_causal_mask(T,device=context_feature.device)

            decoder_pred, pg_pred = self.decoder(context_feature,None, enc_out, encoder_pad_mask, causal_mask,ins_text) # (bs,seq_len,4)

           

            # action sampling
            last_step_logits = decoder_pred[:, -1, :] 
            # action_inferenced = last_step_logits.argmax(dim=-1).unsqueeze(-1)

            # last_step_pg = pg_pred[:, -1, :]
            # if last_step_pg > 0.8:
            #     action_inferenced[0,0] = 0

            probabilities = torch.softmax(last_step_logits, dim=-1)
            probabilities = probabilities.squeeze(1)
            m = torch.distributions.Categorical(probabilities)
            action_inferenced = m.sample().unsqueeze(-1)

            return action_inferenced
            

        
        # encoder
        enc_out = self.instruction_encoder(batch_tokens) # (bs,200,emd)
        enc_out = self.encoder_linear(enc_out)


        # decoder
        context_feature = self.get_context_feature(observations)
        context_feature = context_feature.view(B,T,-1)
        causal_mask = self.generate_causal_mask(T,device=context_feature.device)

        decoder_pred, pred_progress = self.decoder(context_feature,observations["padding_mask"], enc_out, encoder_pad_mask, causal_mask)
        

        action_loss = self.softmax_CE(decoder_pred,observations["gt_actions"].long())
        
    

        if self.config.MODEL.PROGRESS_MONITOR.use:

            progress_loss = F.mse_loss(
                pred_progress,
                observations["progress"],
                reduction="none",
            ).squeeze(-1)


            overall_loss = action_loss + progress_loss
            masked_weighted_loss = MaskedWeightedLoss(overall_loss,observations["lengths"],  observations["weights"]).sum()

        else:
            masked_weighted_loss = MaskedWeightedLoss(action_loss,observations["lengths"],  observations["weights"]).sum()

            
        masked_weighted_loss /= B
        
        return masked_weighted_loss, decoder_pred



    
    def create_boolean_mask(self,mask):
       
        mask = mask.to(torch.int32)
        max_len = mask.max().item()
        expanded_mask = torch.arange(max_len).expand(len(mask), max_len).to(mask.device)
        boolean_mask = expanded_mask < mask.unsqueeze(1)

        return ~boolean_mask 



    def encode_visions(self,batch,config):

        depth_embedding = self.depth_encoder(batch)
        rgb_embedding = self.rgb_encoder(batch)

        return rgb_embedding,depth_embedding
    


