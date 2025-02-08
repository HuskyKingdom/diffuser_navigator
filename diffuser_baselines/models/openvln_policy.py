import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_baselines.rl.ppo.policy import NetPolicy


from diffuser_baselines.models.utils import UnMaskedSoftmaxCELoss,MaskedWeightedLoss

from transformers import BertTokenizer      

from gym import spaces
import numpy as np


# openvln
from transformers import LlamaTokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
# from prismatic.vla.action_tokenizer import ActionTokenizer
from typing import Callable, Dict, List, Optional, Type, Union
from pathlib import Path
from prismatic.models import load
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.utils.rnn import pad_sequence
from PIL import Image



IGNORE_INDEX = -100


@baseline_registry.register_policy
class OpenVLNPolicy(NetPolicy):
    
    def __init__(
        self, config
    ) -> None:
        
        super(NetPolicy, self).__init__()
        self.config = config
        # self.navigator = OpenVLN(norm_stats=None)
        self.rgb_his = []
        self.depth_his = []
        self.pre_actions = None


        # == openvln ==

        # init
        base_vlm = "prism-dinosiglip-224px+7b"
        # base_vlm = "hub/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/prism-dinosiglip-224px+7b/" # find model weight locally

        # load backbones
        hf_token = Path(".hf_token").read_text().strip()
        self.vlm = load(base_vlm, hf_token=hf_token, load_for_training=True)
        self.tokenlizer = self.vlm.llm_backbone.get_tokenizer()
        self.image_transform = self.vlm.vision_backbone.get_image_transform()

        # freeze backbone
        # stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        self.vlm.freeze_backbones(self.config.OPENVLN.stage)

        # initializing special tokens

        if config.OPENVLN.forward_type == "pre-train":
            # <SPC>
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<SPC>"]})
            # actions <FORWARD> <LEFT> <RIGHT> <STOP>
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<FORWARD>"]})
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<LEFT>"]})
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<RIGHT>"]})
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<STOP>"]})
            self.vlm.llm_backbone.llm.resize_token_embeddings(len(self.tokenlizer), pad_to_multiple_of=64)


        

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



        # rgb_features,depth_features = self.navigator.encode_visions(observations,self.config) # stored vision features
        
        

        # format batch data
        collected_data = {
        'instruction': observations['instruction'],
        'rgb': observations['rgb'],

        'gt_actions': observations['gt_actions'], # (bs,T)
        'prev_actions': observations['prev_actions'],
        'trajectories': observations['trajectories'],
        'padding_mask': observations['padding_mask'].bool(),
        'lengths': observations['lengths'],
        'weights': observations['weights'],
        'ins_text': observations['ins_text'],
        'progress': observations['progress'],
        'labels': observations["labels"],
        }


        # == build model input (prompt + label) ==


        # == add <SPC> & tokenlization instructions & labels ==
        for i in range(len(collected_data['ins_text'])):
            collected_data['ins_text'][i] += "<SPC>" 
            collected_data['ins_text'][i] = self.tokenlizer(collected_data['ins_text'][i], truncation=False, return_tensors="pt").input_ids[0] # auto added BOS

        inputids = pad_sequence(collected_data['ins_text'], batch_first=True, padding_value=self.tokenlizer.pad_token_id).to(observations['instruction'].device)
        attention_mask = inputids.ne(self.tokenlizer.pad_token_id)

        

        # for i in range(len(collected_data['labels'])):
        #     collected_data['labels'][i] = self.tokenlizer(collected_data['labels'][i], truncation=False, return_tensors="pt").input_ids[0]
        # inputids_label = pad_sequence(collected_data['labels'], batch_first=True, padding_value=self.tokenlizer.pad_token_id).to(observations['instruction'].device)
        # attention_mask_label = inputids.ne(self.tokenlizer.pad_token_id)




       


        # == formulating images ==

        # # (B,T,H,W,C) -> (B,T,C,H,W)
        # collected_data['rgb'] = collected_data['rgb'].permute(0, 1, 4, 2, 3) 
        

        # reshape and format
        B,T,H,W,C = collected_data['rgb'].shape
        collected_data['rgb'] = collected_data['rgb'].view(-1,H,W,C).cpu().numpy().astype(np.uint8)

        

        # to PIL image for transform
        pil_images = [Image.fromarray(img) for img in collected_data['rgb']]
        transformed_images = [self.image_transform(img) for img in pil_images]

        # back to tensor
        transformed_images_tensor = {
        k: torch.stack([sample[k] for sample in transformed_images]).float().to(observations['instruction'].device)
        for k in transformed_images[0].keys()
        } # in shape {dino: (2120, 3, 224, 224); siglip: (2120, 3, 224, 224)}

        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            modelout = self.vlm(input_ids=inputids, attention_mask=attention_mask,pixel_values=transformed_images_tensor, labels = collected_data['gt_actions'].long(), img_ori_shape = (B,T), sample_valid_len = observations['lengths'], config = self.config)
        

        return modelout.loss
    

    
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




class OpenVLN(PrismaticVLM):


    def __init__(self,
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)
        
        # linear transform vit tokens
        # self.vit_linear = nn.Linear(256,1)

    

    

    def reshape_vision_features(self, projected_patch_embeddings, img_ori_shape):

        pat_dim = projected_patch_embeddings.shape[1]
        feature_dim = projected_patch_embeddings.shape[2]
        projected_patch_embeddings = projected_patch_embeddings.view(img_ori_shape[0], img_ori_shape[1], pat_dim, feature_dim) # (B,T,256,4096)

        # projected_patch_embeddings = self.vit_linear(projected_patch_embeddings)
        projected_patch_embeddings = projected_patch_embeddings.mean(dim=2) # reduce dimension -> (B,T,4096)
        projected_patch_embeddings = projected_patch_embeddings.squeeze(2)

        return projected_patch_embeddings





    def get_input(self, input_ids, img_features, input_mask, img_mask, labels, label_mask):

        multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        # prepare interleaved embedding input =====
        # tokenlize action labels
        token_mapping = torch.tensor([32005, 32002, 32003, 32004]).to(img_features.device)
        tokenlized_labels = token_mapping[labels]
        action_input_embeddings = self.llm_backbone.embed_input_ids(tokenlized_labels)

        

        # interleave actions and observations
        interleaved_tokens = torch.stack((img_features, action_input_embeddings), dim=2).view(img_features.shape[0], img_features.shape[1] * 2, img_features.shape[2])

    

        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :, :],
                interleaved_tokens,
            ],
            dim=1,
        )


        # interleaved mask
        interleaved_mask = label_mask.unsqueeze(2).expand(img_features.shape[0], img_features.shape[1], 2).reshape(img_features.shape[0], img_features.shape[1] * 2)



        multimodal_attention_mask = None
        multimodal_attention_mask = torch.cat(
                [
                    input_mask[multimodal_indices, :],
                    interleaved_mask,
                ],
                dim=1,
        )


        # label indicator
        img_labels = torch.full(
                (tokenlized_labels.shape[0], tokenlized_labels.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
        interleaved_labels = torch.stack((img_labels, tokenlized_labels), dim=2).view(tokenlized_labels.shape[0], tokenlized_labels.shape[1] * 2)

       
        return multimodal_embeddings, multimodal_attention_mask, interleaved_labels
        
     
                

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        img_ori_shape = None,
        sample_valid_len = None,
        config = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLN, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # ===== training forward logic =====



        # ==== IMG PATCH FEATURES ====
        # Run Visual Feature Extraction # in shape {dino: (2120, 3, 224, 224); siglip: (2120, 3, 224, 224)}
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                # patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
                patch_features = self.vision_backbone({k: pixel_values[k] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])
        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)

        # ==== FORMULATE IMG FEATURES ====
        # openvln: reshape the resulting features into seq shape & generate img attention mask
        img_pat_feature = self.reshape_vision_features(projected_patch_embeddings, img_ori_shape) # (bs,T,4096)
        
        # ==== IMG PATCH MASK ====
        img_pat_mask = torch.arange(img_ori_shape[1]).unsqueeze(0).to(img_pat_feature.device) < sample_valid_len.unsqueeze(1) # (B,T)

        img_pat_mask = img_pat_mask.to(
            dtype=attention_mask.dtype,
            device=attention_mask.device,
            )


        
        # ==== INPUT & MASK ====
        multimodal_embeddings, multimodal_attention_mask, interleaved_labels = self.get_input(input_ids=input_ids,img_features=img_pat_feature, input_mask=attention_mask, img_mask=img_pat_mask, labels=labels, label_mask=img_pat_mask)



        # ==== LABEL ====
        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!


        multimodal_labels = None
        if labels is not None:

            # projected_patch_labels = torch.full(
            #     (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            #     IGNORE_INDEX,
            #     dtype=labels.dtype,
            #     device=labels.device,
            # )


            input_labels = torch.full(
                (input_ids.shape[0], input_ids.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            multimodal_labels = torch.cat(
                [input_labels, interleaved_labels], dim=1
            )


        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!

        
        return self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask, # ([3, 128])
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings, # ([3, 128, 4096])
            labels=fused_labels, # ([3, 146])
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )





