import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.common.baseline_registry import baseline_registry
import re
from habitat_baselines.rl.ppo.policy import NetPolicy


from diffuser_baselines.models.utils import UnMaskedSoftmaxCELoss,MaskedWeightedLoss

from transformers import BertTokenizer      

from gym import spaces
import numpy as np
import random

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
from prismatic.util.nn_utils import FusedMLPProjector
from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule
from diffuser_baselines.models.common.position_encodings import PositionalEncoding
from diffuser_baselines.models.utils import MemoryLlamaDecoderLayer, NormalLlamaDecoderLayer, MemoryPhiDecoderLayer
from transformers.loss.loss_utils import fixed_cross_entropy


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
        self.pre_actions = None
        self.addti_action_token = None


        # == openvln ==
        # init backbone
        base_vlm = "phi-2+3b" # prism-dinosiglip-224px+7b
        replace_layer = MemoryPhiDecoderLayer

        # load backbones
        hf_token = Path(".hf_token").read_text().strip()
        self.vlm = load(base_vlm, hf_token=hf_token, load_for_training=True, flash_atten = config.OPENVLN.flash_atten)
        self.tokenlizer = self.vlm.llm_backbone.get_tokenizer()
        self.image_transform = self.vlm.vision_backbone.get_image_transform()

        # freeze backbone
        # stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        self.vlm.freeze_backbones(self.config.OPENVLN.stage)


        if config.OPENVLN.forward_type == "pre-train":
            # <SPC>
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<SPC>"]})
            # actions <FORWARD> <LEFT> <RIGHT> <STOP>
            # self.tokenlizer.add_tokens(["<FORWARD>","<LEFT>","<RIGHT>","<STOP>"])
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<FORWARD>"]})
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<LEFT>"]})
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<RIGHT>"]})
            self.tokenlizer.add_special_tokens({"additional_special_tokens": ["<STOP>"]})
            # self.vlm.llm_backbone.llm.resize_token_embeddings(len(self.tokenlizer), pad_to_multiple_of=64)
            self.vlm.llm_backbone.llm.resize_token_embeddings(len(self.tokenlizer))


        original_layers = self.vlm.llm_backbone.llm.model.layers
        new_decoder_layers = nn.ModuleList()
        total_layers = len(original_layers)

        for idx, old_layer in enumerate(original_layers):
            new_layer = replace_layer(self.vlm.llm_backbone.llm.model.config, layer_idx=idx)
            new_layer.load_state_dict(old_layer.state_dict(), strict=False)
            new_decoder_layers.append(new_layer)

        self.vlm.llm_backbone.llm.model.layers = new_decoder_layers

        del original_layers

      


    def formating_input_frame(self,current_frame,device):

        if not current_frame.is_contiguous():
            current_frame = current_frame.contiguous()

        current_frame = current_frame.detach().cpu().numpy().astype(np.uint8)

        # to PIL image for transform
        pil_images = [Image.fromarray(img) for img in current_frame]
        transformed_images = [self.image_transform(img) for img in pil_images]

        # back to tensor
        if isinstance(transformed_images[0], dict):
            transformed_his_tensor = {
                k: torch.stack([sample[k] for sample in transformed_images]).float().to(device)
                for k in transformed_images[0].keys()
                } # in shape {dino: (2120, 3, 224, 224); siglip: (2120, 3, 224, 224)}
        else:
            transformed_images_tensor = torch.stack([sample for sample in transformed_images]).float().to(device) # (his_len, 3, 336, 336)
        

        return transformed_images_tensor
        

    def formating_history_frames(self,full_histories,device):

        B,T,H,W,C = full_histories.shape

        # appfront init memory
        init_empty_memory = torch.zeros(B,1,H,W,C).to(full_histories.device)
        full_histories = torch.cat((init_empty_memory,full_histories), dim = 1)
        B,T,H,W,C = full_histories.shape

        # reshape and format
        if not full_histories.is_contiguous():
            full_histories = full_histories.contiguous()
        full_histories = full_histories.view(-1,H,W,C).detach().cpu().numpy().astype(np.uint8)


        # to PIL image for transform
        pil_images_his = [Image.fromarray(img) for img in full_histories]
        transformed_his = [self.image_transform(img) for img in pil_images_his]

        # back to tensor
        if isinstance(transformed_his[0], dict):
            transformed_his_tensor = {
                k: torch.stack([sample[k] for sample in transformed_his]).float().to(device)
                for k in transformed_his[0].keys()
                } # in shape {dino: (his_len, 3, 224, 224); siglip: (his_len, 3, 224, 224)}
        else:
            transformed_his_tensor = torch.stack([sample for sample in transformed_his]).float().to(device) # (his_len, 3, 336, 336)



        return transformed_his_tensor,(B,T)




        

    def act(self,observations, prev_actions, encode_only=False,print_info = False,ins_text=None): 

       
        rgb = observations["rgb"]

        # storing histories
        self.rgb_his.append(rgb)


        # action inference
        rgbs = torch.stack(self.rgb_his, dim=1) # (1,seq,224,224,3)


        # print(ins_text)
       
        # format batch data
        collected_data = {
        'rgb': rgbs,
        'ins_text': [ins_text],
        'lengths': torch.tensor([rgbs.shape[1]])
        }


        # build prompt
        for sample in range(len(collected_data['ins_text'])):
            self.prompt_builder = self.vlm.get_prompt_builder()
            self.prompt_builder.add_turn(role="human", message=f"Which action should the robot take now to {collected_data['ins_text'][sample]}?")
            prompt_text = self.prompt_builder.get_prompt()
            collected_data['ins_text'][sample] = self.tokenlizer(prompt_text, truncation=False, return_tensors="pt").input_ids[0] # auto added BOS , in shape (T)
    
        inputids = pad_sequence(collected_data['ins_text'], batch_first=True, padding_value=self.tokenlizer.pad_token_id).to(observations['instruction'].device)



        # == formulating images == (single frame)
        current_frame = collected_data['rgb'][:,-1:,:,:,:]
        current_frame = current_frame.squeeze(1)
        transformed_images_tensor = self.formating_input_frame(current_frame,observations['instruction'].device)
        

        # == prepare histories ==
        full_histories = collected_data['rgb'][:,:-1,:,:,:]
        transformed_his_tensor,img_ori_shape = self.formating_history_frames(full_histories,observations['instruction'].device)




        start_idx = -1 # indicates no his_masking needed

        cast_type = torch.float16
        if self.config.OPENVLN.flash_atten and not self.config.OPENVLN.phase == "phi":
            cast_type = torch.bfloat16

        with torch.cuda.amp.autocast(dtype=cast_type):
            modelout = self.vlm(input_ids=inputids, attention_mask=None,pixel_values=transformed_images_tensor, labels = None, img_ori_shape = img_ori_shape, sample_valid_len = collected_data['lengths'], inference = True, full_his = transformed_his_tensor, sample_start = start_idx)
        


        # retrive action prediction
        inference_logits = modelout.logits[-1]

        # retrive last action logits
        predicted_token_id = torch.argmax(modelout.logits, dim=-1)
        predicted_token_id_list = predicted_token_id.cpu().tolist()
        decoded_tokens = self.tokenlizer.convert_ids_to_tokens(predicted_token_id_list[0])
        # decoded_text = self.tokenlizer.decode(predicted_token_id_list[0], skip_special_tokens=False)

    
        action_token = decoded_tokens[-1]

        if action_token == "<LEFT>":
            action = [[2]]
        elif action_token == "<RIGHT>":
            action = [[3]]
        elif action_token == "<FORWARD>":
            action = [[1]]
        elif action_token == "<STOP>":
            action = [[0]]
        else:
            action = [[0]]

        # past long episodes
        if len(self.rgb_his) >= 300:
            action = [[0]]

        action = torch.tensor(action).to(modelout.logits.device)

        if print_info:
            print(f"Action inferenced : {action[0][0]}, with history length {len(self.rgb_his)}; tokens len {len(decoded_tokens)}")
            # inst_len = collected_data['ins_text'].shape[1]
            # print(decoded_tokens)


        return action
        
    

    def build_loss(self,observations):



        # format batch data
        collected_data = {
        'instruction': observations['instruction'],
        'rgb': observations['rgb'],
        'rgb_prev': observations['rgb_prev'],
        'rgb_prev_mask': observations['rgb_prev_mask'],
        'gt_actions': observations['gt_actions'], # (bs,T)
        'weights': observations['weights'],
        'ins_text': observations['ins_text'],
        'labels': observations["labels"],
        }

   
        # == build model input (prompt + label) ==

        # == add <SPC> & tokenlization instructions & labels ==
        for sample in range(len(collected_data['ins_text'])):
            # build prompt
            self.prompt_builder = self.vlm.get_prompt_builder()
            self.prompt_builder.add_turn(role="human", message=f"Which action should the robot take now to {collected_data['ins_text'][sample]}?")
            prompt_text = self.prompt_builder.get_prompt()
            
            prompt_text += collected_data["labels"][sample]
            collected_data['ins_text'][sample] = self.tokenlizer(prompt_text, truncation=False, return_tensors="pt").input_ids[0] # auto added BOS , in shape (T)
    
        inputids = collected_data['ins_text']
        inputids = pad_sequence(inputids, batch_first=True, padding_value=self.tokenlizer.pad_token_id).to(observations['instruction'].device)
        

        attention_mask = inputids.ne(self.tokenlizer.pad_token_id)
        last_valid_index = attention_mask.sum(dim=-1) - 1 # for indicating the groundtruth action

  
        # == formulating training images ==
        # to PIL image for transform
        transformed_images_tensor = self.formating_input_frame(collected_data['rgb'],observations['instruction'].device)
        

        # == formulating histories ==
        # to PIL image for transform
        full_histories = collected_data['rgb_prev']
        transformed_his_tensor,img_ori_shape = self.formating_history_frames(full_histories,observations['instruction'].device)

        B = img_ori_shape[0]
        # appfront init memory & update history mask
        init_empty_memory_mask = torch.zeros(B,1).bool().to(full_histories.device)
        collected_data['rgb_prev_mask'] = torch.cat((init_empty_memory_mask,collected_data['rgb_prev_mask']), dim = 1)




        # == formulating labels ==
        input_labels = inputids.detach().clone() # create labels

        cast_type = torch.float16
        if self.config.OPENVLN.flash_atten and not self.config.OPENVLN.phase == "phi":
            cast_type = torch.bfloat16

 

        with torch.cuda.amp.autocast(dtype=cast_type):
            modelout = self.vlm(input_ids=inputids, attention_mask=attention_mask,pixel_values=transformed_images_tensor, labels = input_labels, img_ori_shape = img_ori_shape, sample_valid_len = last_valid_index, full_his = transformed_his_tensor, pre_mask = collected_data['rgb_prev_mask'], vocab_size=len(self.tokenlizer), inf_weights = collected_data['weights'])
        

        pred = modelout.logits.argmax(dim=-1)
        predicted_token_id_list = pred.cpu().tolist() # (bs,token_len)

        bs_result = []
        for i in range(len(predicted_token_id_list)):
            decoded_tokens = self.tokenlizer.convert_ids_to_tokens(predicted_token_id_list[i])
            # retrive model prediction for action
            prev_token = last_valid_index - 1
            bs_result.append(decoded_tokens[prev_token[i]])

        print(collected_data['gt_actions'],bs_result)


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
        self.pre_actions = None




class OpenVLN(PrismaticVLM):


    def __init__(self,
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

        hidden_dim = 2560 # 4096
        
        # initialize memory bank
        self.menmory_embedding = nn.Embedding(128,hidden_dim)
        self.M_init = self.menmory_embedding.weight # referencing copy, this will also be updated while loading pre-trained weights

        
        self.memory_fuser_attention = FFWRelativeCrossAttentionModule(hidden_dim,2,1)
        self.pe_layer = PositionalEncoding(hidden_dim,0.2)






    def get_input(self, input_ids, img_features, input_mask, labels,valid_len = None):
        
        
        # ====== Encode InputIDS ======
        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
    

        # ====== Generate InputEmbeddings ======
        # input embeddings in shape <bos> + img_pathes + input + label
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[:, :1, :],
                img_features,
                input_embeddings[:, 1:, :]
            ],
            dim=1,
        )

        
        # interleaved mask
        multimodal_attention_mask = None
        if input_mask != None:
            projectoed_patch_attention_mask = torch.full(
                    (img_features.shape[0], img_features.shape[1]),
                    True,
                    dtype=labels.dtype,
                    device=labels.device,
                )
            multimodal_attention_mask = torch.cat(
                    [
                        input_mask[:, :1],
                        projectoed_patch_attention_mask,
                        input_mask[:, 1:]
                    ],
                    dim=1,
            )



        # label indicator
        multimodal_labels = None
        if input_mask != None:
            

            bs = labels.size(0)
            new_labels = torch.full_like(labels, IGNORE_INDEX)
            indices = torch.arange(bs)
            new_labels[indices, valid_len] = labels[indices, valid_len]


            projectoed_patch_labels = torch.full(
                    (img_features.shape[0], img_features.shape[1]),
                    IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )

            multimodal_labels = torch.cat(
                    [
                        new_labels[:,:1],
                        projectoed_patch_labels,
                        new_labels[:,1:],
                    ],
                    dim=1,
            )


        return multimodal_embeddings, multimodal_attention_mask, multimodal_labels
        
    
    def inference_action(self,input_ids, pixel_values, full_his, img_ori_shape,sample_valid_len):

        # ===== inference forward logic =====

        # ==== IMG PATCH FEATURES ====
        # Run Visual Feature Extraction # in shape {dino: (2120, 3, 224, 224); siglip: (2120, 3, 224, 224)}
        patch_features = self.vision_backbone({k: pixel_values[k] for k in pixel_values})

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)

       
        # ==== INPUT & MASK ====
        multimodal_embeddings, _ , _ = self.get_input(input_ids=input_ids,img_features=projected_patch_embeddings, input_mask=None, labels=None)


        # ==== Update Memories ====
        # express histories in [cls] way
        full_his_patches = self.vision_backbone({k: full_his[k] for k in full_his}) # (bs*T,vit_token_len,dim)

        projected_his_embeddings = self.projector(full_his_patches)
        projected_cls_embeddings = self.extract_cls(projected_his_embeddings) # (bs*T,4,dim)
        

        compressed_memory = self.compress_memories(projected_cls_embeddings,img_ori_shape,multimodal_embeddings,pre_mask = None)


        inference_result = self.llm_backbone(
            input_ids=None,
            attention_mask=None, # inference
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings, # ([3, 128, 4096])
            labels=None, # inference
            compressed_mem=compressed_memory,
        )


        return inference_result


    def extract_cls(self,projected_patch_embeddings): # grid pooling, following Navid. His_len = bs * max_his_len

        if not projected_patch_embeddings.is_contiguous():
            projected_patch_embeddings = projected_patch_embeddings.contiguous()
        
        His_len,Token_len,C = projected_patch_embeddings.shape

        if Token_len == 256:
            _grid = projected_patch_embeddings.view(His_len,16,16,C)
        else:
            _grid = projected_patch_embeddings.view(His_len,24,24,C)

        _grid = _grid.permute(0,3,1,2)
        pool = nn.AdaptiveAvgPool2d((2,2))
        pooled = pool(_grid)

        result = pooled.permute(0,2,3,1).reshape(His_len,4,C)

        return result


    def compress_memories(self,projected_cls_embeddings,img_ori_shape,multimodal_embeddings,pre_mask): # grid pooled version

        if not projected_cls_embeddings.is_contiguous():
            projected_cls_embeddings = projected_cls_embeddings.contiguous() # (his_len,4,dim)
        if not self.M_init.is_contiguous():
            self.M_init = self.M_init.contiguous() 

        

        token_bs = multimodal_embeddings.shape[0]
        his_T,grid_len,hidden_dim = projected_cls_embeddings.shape

        projected_cls_embeddings = projected_cls_embeddings.view(img_ori_shape[0],img_ori_shape[1],grid_len,hidden_dim)
        projected_cls_embeddings = projected_cls_embeddings.view(img_ori_shape[0],-1,hidden_dim)  # (bs,his_len * 4,dim)

     
        his_pos = self.pe_layer(projected_cls_embeddings)


        # format init memory
        # resulting memory in (token_bs,C,d)
        expanded_memory = self.M_init.unsqueeze(0).expand(token_bs,-1,-1)


        # repeat 4 times on his_T dimension for grid pooled version
        pre_mask = pre_mask.repeat_interleave(4,dim=1)
        pre_mask = pre_mask.bool()

        

        # compressing
        compressed_memory,_ = self.memory_fuser_attention(query=expanded_memory.transpose(0, 1),
            value=his_pos.transpose(0, 1),
            query_pos=None,
            value_pos=his_pos,
            diff_ts=None,pad_mask=pre_mask,print_info=False)
        compressed_memory = compressed_memory[-1].transpose(0,1) # (bs*T,C,d)


    
        return compressed_memory
    



    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # expect labels during train, expect prev_actions during inference
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        img_ori_shape = None,
        sample_valid_len = None,
        inference: Optional[bool] = False,
        full_his = None,
        pre_mask = None,
        vocab_size = None,
        inf_weights = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLN, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)

        if inference:
            return self.inference_action(input_ids, pixel_values, full_his, img_ori_shape,sample_valid_len)


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
                patch_features = self.vision_backbone(pixel_values)
        
        

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)



        # ==== INPUT & LABEL & MASK ====
        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_embeddings, multimodal_attention_mask, multimodal_labels = self.get_input(input_ids=input_ids,img_features=projected_patch_embeddings, input_mask=attention_mask, labels=labels, valid_len=sample_valid_len)

    

        # ==== Update Memories ====
        # express history tokens by grid pool

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(full_his, dict):
                full_his_patches = self.vision_backbone({k: full_his[k] for k in full_his}) # (bs*T,vit_token_len,dim)
            else:
                full_his_patches = self.vision_backbone(full_his)
        

        projected_his_embeddings = self.projector(full_his_patches)
        projected_cls_embeddings = self.extract_cls(projected_his_embeddings) # (bs*T,4,dim)
        
        compressed_memory = self.compress_memories(projected_cls_embeddings,img_ori_shape,multimodal_embeddings,pre_mask)
        

        
        
        # Run LLM Forward --> returns CausalLMOutputWithPast!


        out = self.llm_backbone(
            input_ids=None,
            attention_mask=multimodal_attention_mask, # ([bs*T, token_len])
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=multimodal_embeddings, # ([bs*T, token_len, dim])
            labels=multimodal_labels, # ([bs*T, token_len])
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            compressed_mem=compressed_memory, # ([bs*T, C, d])
        )
        
        
        # calculating loss
        logits = out.logits.float() # (bs,token_len,dim)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = multimodal_labels[..., 1:].contiguous()

        print(f"Forward Print {shift_logits[:,-1,-6:]};")

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = nn.functional.cross_entropy(shift_logits,shift_labels,ignore_index=-100,reduction="none")

        # loss weighting
        bs,seq = multimodal_labels[..., 1:].size()
        loss = loss.contiguous()
        inf_weights = inf_weights.contiguous()
        
        loss = loss.view(bs,seq)
        inf_weights = inf_weights.unsqueeze(1)
        weighted_loss = loss * inf_weights

        final_loss = weighted_loss.sum() / bs

        
      
        out.loss = final_loss
        
        
            
        return out





