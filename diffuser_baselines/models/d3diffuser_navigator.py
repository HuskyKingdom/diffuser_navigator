import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from habitat_baselines.common.baseline_registry import baseline_registry

from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule, FFWRelativeSelfAttentionModule,ParallelAttention
from habitat_baselines.rl.ppo.policy import Policy
from diffuser_baselines.models.encoders.instruction_encoder import InstructionEncoder
from diffuser_baselines.models.encoders.trajectory_decoder import TrajectoryDecoder

from diffuser_baselines.models.common.position_encodings import RotaryPositionEncoding,SinusoidalPosEmb, PositionalEncoding

from diffuser_baselines.models.encoders import resnet_encoders
from diffuser_baselines.models.encoders.his_encoder import HistoryGRU
from diffuser_baselines.models.utils import MaskedSoftmaxCELoss

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
        self.pre_actions = []
        

    def act(self,observations, prev_actions, encode_only=False,print_info = False,ins_text=None): 

        
        rgb_features,depth_features = self.navigator.encode_visions(observations,self.config) # raw batch

        # storing histories
        self.rgb_his.append(rgb_features)
        self.depth_his.append(depth_features)
        if prev_actions.item() != 0:
            self.pre_actions.append(prev_actions.item())

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
        'prev_actions': torch.tensor(self.pre_actions).unsqueeze(0).to(observations['instruction'].device).long(),
        'trajectories': None,
        'padding_mask': None,
        'lengths': None,
        'weights': None,
        'ins_text': [ins_text]
        }

        
        action = self.navigator(collected_data,(B,T), inference = True, ins_text=ins_text)
        

        if print_info:
            print(f"Action inferenced : {action.item()}, with history length {len(self.pre_actions)}")


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
        'ins_text': observations['ins_text']
        }

        
        loss = self.navigator(collected_data,(B,T), inference = False)


        return loss
    
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
        self.pre_actions = []




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

        self.depth_encoder.to(next(self.parameters()).device).train()
        self.rgb_encoder.to(next(self.parameters()).device).train()

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

        self.masked_CE = MaskedSoftmaxCELoss()


        # self.time_emb = nn.Sequential(
        #     SinusoidalPosEmb(embedding_dim),
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, embedding_dim)
        # )

        # self.his_encoder = HistoryGRU(32768,512,embedding_dim,2)

        # # positional embeddings
        # self.pe_layer = PositionalEncoding(embedding_dim,0.2)

        # # Attention layers _____________________
        # layer = ParallelAttention(
        #     num_layers=num_layers,
        #     d_model=embedding_dim, n_heads=num_attention_heads,
        #     self_attention1=False, self_attention2=False,
        #     cross_attention1=True, cross_attention2=False
        # )
        # self.vl_attention = nn.ModuleList([
        #     layer
        #     for _ in range(1)
        #     for _ in range(1)
        # ])
        # self.traj_lang_attention = nn.ModuleList([
        #     ParallelAttention(
        #         num_layers=1,
        #         d_model=embedding_dim, n_heads=num_attention_heads,
        #         self_attention1=False, self_attention2=False,
        #         cross_attention1=True, cross_attention2=False,
        #         rotary_pe=False, apply_ffn=False
        #     )
        # ])

        # self.history_self_atten = FFWRelativeSelfAttentionModule(embedding_dim,2,1)
        # self.action_self_atten = FFWRelativeSelfAttentionModule(embedding_dim,2,1)
        # self.language_self_atten = FFWRelativeSelfAttentionModule(embedding_dim,num_attention_heads,num_layers)

        # self.language_his_cross_atten = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)
        # self.cross_attention = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)
        # self.context_cross_attention = FFWRelativeCrossAttentionModule(embedding_dim,num_attention_heads,num_layers)

        # # predictors

        # self.sample_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, self.config.DIFFUSER.traj_space)
        # )

        # self.n_steps = diffusion_timesteps
    



    def get_context_feature(self,observations,inference=False):

        

        bs = observations["instruction"].size(0)

        rgb_features = self.rgb_linear(observations["rgb_features"])  # (B+T, channel, 4,4) -> (B+T, emb)
        depth_features = self.depth_linear(observations["depth_features"]) # (B+T, channel, 1,1) -> (B+T, emb)

        if not inference: # compute action featrues based on gt actions

            # construct input as [<start>,...]
            # action_except = observations['gt_actions'][:, :-1] # remove last element <end>
            action_except = observations['prev_actions'][:, 1:] # remove first
            action_start_token = torch.full((observations['prev_actions'].shape[0], 1), 4).to(observations['prev_actions'].device) # add start token
            action_input = torch.cat([action_start_token, action_except], dim=1) # construct input
            action_input = action_input.view(-1,) # # (B,T) -> (B+T,)
            action_features = self.action_encoder(action_input.long()) # (B+T,) -> (B+T, emb)

        else: # compute action featrues based on decoder outputs (prev actions)
            
            print( observations['prev_actions'], observations['prev_actions'].shape)
            assert 1==2
            action_start_token = torch.full((observations['prev_actions'].shape[0], 1), 4).to(observations['prev_actions'].device) # add start token
            action_input = torch.cat([action_start_token, observations['prev_actions']], dim=1) # construct input
            action_input = action_input.view(-1,) # # (B,T) -> (B+T,)
            action_features = self.action_encoder(action_input.long()) # (B+T,) -> (B+T, emb)




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

            decoder_pred = self.decoder(context_feature,None, enc_out, encoder_pad_mask, causal_mask,ins_text) # (bs,seq_len,4)

            # action sampling
            last_step_logits = decoder_pred[:, -1, :] 
            action_inferenced = last_step_logits.argmax(dim=-1).unsqueeze(-1)
            # probabilities = torch.softmax(last_step_logits, dim=-1)
            # probabilities = probabilities.squeeze(1)
            # m = torch.distributions.Categorical(probabilities)
            # action_inferenced = m.sample().unsqueeze(-1)

            return action_inferenced
            

        
        # encoder
        enc_out = self.instruction_encoder(batch_tokens) # (bs,200,emd)
        enc_out = self.encoder_linear(enc_out)


        # decoder
        context_feature = self.get_context_feature(observations)
        context_feature = context_feature.view(B,T,-1)
        causal_mask = self.generate_causal_mask(T,device=context_feature.device)

        decoder_pred = self.decoder(context_feature,observations["padding_mask"], enc_out, encoder_pad_mask, causal_mask)


        loss = self.masked_CE(decoder_pred,observations["gt_actions"].long(), observations["lengths"],  observations["weights"]).sum()
        loss /= B


        return loss



    def vision_language_attention(self, feats, instr_feats,seq1_pad=None,seq2_pad=None):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=seq1_pad,
            seq2=instr_feats, seq2_key_padding_mask=seq2_pad,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats
    
    
    def create_boolean_mask(self,mask):
       
        mask = mask.to(torch.int32)
        max_len = mask.max().item()
        expanded_mask = torch.arange(max_len).expand(len(mask), max_len).to(mask.device)
        boolean_mask = expanded_mask < mask.unsqueeze(1)

        return ~boolean_mask 


    def predict_logits(self, x, tokens, timesteps): # tokens in form [instr_tokens,rgb_tokens,depth_tokens,history_tokens,pad_mask, his_len]

        time_embeddings = self.time_emb(timesteps.float())
        pad_mask = tokens[4]
        his_pad_mask = self.create_boolean_mask(tokens[5])

        # encode actions
        action_emb = self.action_encoder(x)

  
        
        # positional embedding __________________
        instruction_position = self.pe_layer(tokens[0])
        action_position = self.pe_layer(action_emb)
        history_position = self.pe_layer(tokens[3])

        # self attentions __________________
        # encode history
        history_feature = self.history_self_atten(history_position.transpose(0,1), diff_ts=time_embeddings,
                query_pos=None, context=None, context_pos=None,pad_mask=his_pad_mask)[-1].transpose(0,1)
        

        # action features
        action_features = self.action_self_atten(action_position.transpose(0,1), diff_ts=time_embeddings,
                query_pos=None, context=None, context_pos=None,pad_mask=None)[-1].transpose(0,1)

        # languege features
        lan_features = self.language_self_atten(instruction_position.transpose(0,1), diff_ts=time_embeddings,
                query_pos=None, context=None, context_pos=None,pad_mask=pad_mask)[-1].transpose(0,1)
        

        # cross attentions __________________
        lan_features = self.language_his_cross_atten(query=lan_features.transpose(0, 1),
            value=history_feature.transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings,pad_mask=his_pad_mask)[-1].transpose(0,1) # takes last layer and return to (B,L,D)

        # observation features
        obs_features = self.cross_attention(query=tokens[1].transpose(0, 1),
            value=tokens[2].transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings)[-1].transpose(0,1) # takes last layer and return to (B,L,D)
        
        # context features 
        context_features = self.vision_language_attention(obs_features,lan_features,seq2_pad=pad_mask) # rgb attend instr.


        # prediction head
        final_feature = self.context_cross_attention(query=action_features.transpose(0, 1),
            value=context_features.transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=time_embeddings)[-1].transpose(0,1) 

        logits_pred = self.sample_predictor(final_feature)

        return logits_pred 


    def encode_visions(self,batch,config):

        depth_embedding = self.depth_encoder(batch)
        rgb_embedding = self.rgb_encoder(batch)

        return rgb_embedding,depth_embedding
    





class D3PM(nn.Module):
    def __init__(
        self,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 10,
        forward_type="uniform",
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(D3PM, self).__init__()
        self.x0_model = x0_model

        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        # self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-6
        self.num_classses = num_classes
        q_onestep_mats = []
        q_mats = []  # these are cumulative

        for beta in self.beta_t:

            if forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        q_one_step_transposed = q_one_step_mats.transpose(
            1, 2
        )  # this will be used for q_posterior_logits

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.logit_type = "logit"

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.q_mats.shape

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classses,), print(
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"
        )

        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def vb(self, dist1, dist2):

        # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x_0, t, cond):
        # this part exists because in general, manipulation of logits from model's logit
        # so they are in form of x_0's logit might be independent to model choice.
        # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
        # they introduce at appendix A.8.

        predicted_x0_logits = self.x0_model(x_0, t, cond)

        return predicted_x0_logits

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
        x is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
        """
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        x_t = self.q_sample(
            x, t, torch.rand((*x.shape, self.num_classses), device=x.device)
        )

        # x_t is same shape as x
        assert x_t.shape == x.shape, print(
            f"x_t.shape: {x_t.shape}, x.shape: {x.shape}"
        )
        # we use hybrid loss.

        predicted_x0_logits = self.model_predict(x_t, t, cond)

        # based on this, we first do vb loss.
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)

        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)

        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * vb_loss + ce_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, noise):

        predicted_x0_logits = self.model_predict(x, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def sample(self, x, cond=None):
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )

        return x

    def sample_with_image_sequence(self, x, cond=None, stride=10):
        steps = 0
        images = []
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device)
            )
            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
