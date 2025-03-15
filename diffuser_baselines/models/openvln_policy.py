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
from diffuser_baselines.models.utils import MemoryLlamaDecoderLayer, NormalLlamaDecoderLayer
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

        # init
        base_vlm = "prism-dinosiglip-224px+7b"
        # base_vlm = "hub/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/prism-dinosiglip-224px+7b/" # find model weight locally

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
            # if idx >= total_layers - 10:
            #     new_layer = MemoryLlamaDecoderLayer(self.vlm.llm_backbone.llm.model.config, layer_idx=idx)
            # else:
            #     new_layer = NormalLlamaDecoderLayer(self.vlm.llm_backbone.llm.model.config, layer_idx=idx)
            new_layer = MemoryLlamaDecoderLayer(self.vlm.llm_backbone.llm.model.config, layer_idx=idx)
            new_layer.load_state_dict(old_layer.state_dict(), strict=False)
            new_decoder_layers.append(new_layer)

        self.vlm.llm_backbone.llm.model.layers = new_decoder_layers

        del original_layers

      




        

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
        # reshape and format
        current_frame = collected_data['rgb'][:,-1:,:,:,:]

        B,T,H,W,C = current_frame.shape
        current_frame = current_frame.view(-1,H,W,C).cpu().numpy().astype(np.uint8)

        # to PIL image for transform
        pil_images = [Image.fromarray(img) for img in current_frame]
        transformed_images = [self.image_transform(img) for img in pil_images]

        # back to tensor
        transformed_images_tensor = {
        k: torch.stack([sample[k] for sample in transformed_images]).float().to(observations['rgb'].device)
        for k in transformed_images[0].keys()
        } # in shape {dino: (1, 3, 224, 224); siglip: (1, 3, 224, 224)}
        


        # == prepare histories ==
        full_histories = collected_data['rgb'][:,:-1,:,:,:]
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
        transformed_his_tensor = {
        k: torch.stack([sample[k] for sample in transformed_his]).float().to(observations['instruction'].device)
        for k in transformed_his[0].keys()
        } # in shape {dino: (2120, 3, 224, 224); siglip: (2120, 3, 224, 224)}

        start_idx = -1 # indicates no his_masking needed

        cast_type = torch.float16
        if self.config.OPENVLN.flash_atten:
            cast_type = torch.bfloat16

        with torch.cuda.amp.autocast(dtype=cast_type):
            modelout = self.vlm(input_ids=inputids, attention_mask=None,pixel_values=transformed_images_tensor, labels = None, img_ori_shape = (B,T), sample_valid_len = collected_data['lengths'], inference = True, full_his = transformed_his_tensor, sample_start = start_idx)
        


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


        if self.config.OPENVLN.truncation: # current version only supports local bs = 1
            # truncats batch size to preventing cuda OOM
            total_ts = observations['rgb'].shape[1]
            truncation_len = 10
            if total_ts >= truncation_len:
                start_idx = random.randint(0, total_ts - truncation_len)
                end_idx = start_idx + truncation_len
            else:
                start_idx = 0
                end_idx = total_ts

            collected_data['rgb'] = observations['rgb'][:, start_idx:end_idx, :, :, :]
            collected_data['gt_actions'] = observations['gt_actions'][:, start_idx:end_idx]
            collected_data['weights'] = observations['weights'][:,start_idx:end_idx]
            collected_data['ins_text'] = [row[start_idx:end_idx] for row in observations['ins_text']]
            collected_data['labels'] = [row[start_idx:end_idx] for row in observations['labels']]
            full_histories = observations['rgb'].clone()

            

        # == build model input (prompt + label) ==

        # == add <SPC> & tokenlization instructions & labels ==
        for sample in range(len(collected_data['ins_text'])):
            for ts in range(len(collected_data['ins_text'][sample])):
                # build prompt
                self.prompt_builder = self.vlm.get_prompt_builder()

                # conversation = [
                #     {"from": "human", "value": f"Which action should the robot take now to {collected_data['ins_text'][sample][ts]}"},
                #     {"from": "gpt", "value": collected_data["labels"][sample][ts]},
                # ]
                # for turn in conversation:
                #     self.prompt_builder.add_turn(turn["from"], turn["value"])
                # print(self.prompt_builder.get_prompt())
                # assert 1==2
               

                self.prompt_builder.add_turn(role="human", message=f"Which action should the robot take now to {collected_data['ins_text'][sample][ts]}?")
                prompt_text = self.prompt_builder.get_prompt()
                
                prompt_text += collected_data["labels"][sample][ts]
                collected_data['ins_text'][sample][ts] = self.tokenlizer(prompt_text, truncation=False, return_tensors="pt").input_ids[0] # auto added BOS , in shape (T)
        
        inputids = [item for sublist in collected_data['ins_text'] for item in sublist]
        inputids = pad_sequence(inputids, batch_first=True, padding_value=self.tokenlizer.pad_token_id).to(observations['instruction'].device)
        

        attention_mask = inputids.ne(self.tokenlizer.pad_token_id)

        

        # == formulating training images ==
        # reshape and format
        B,T,H,W,C = collected_data['rgb'].shape
        if not collected_data['rgb'].is_contiguous():
            collected_data['rgb'] = collected_data['rgb'].contiguous()
        collected_data['rgb'] = collected_data['rgb'].view(-1,H,W,C).detach().cpu().numpy().astype(np.uint8)

        


        # to PIL image for transform
        pil_images = [Image.fromarray(img) for img in collected_data['rgb']]
        transformed_images = [self.image_transform(img) for img in pil_images]


        # back to tensor
        transformed_images_tensor = {
        k: torch.stack([sample[k] for sample in transformed_images]).float().to(observations['instruction'].device)
        for k in transformed_images[0].keys()
        } # in shape {dino: (2120, 3, 224, 224); siglip: (2120, 3, 224, 224)}
        

        # == formulating histories ==
       
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
        transformed_his_tensor = {
        k: torch.stack([sample[k] for sample in transformed_his]).float().to(observations['instruction'].device)
        for k in transformed_his[0].keys()
        } # in shape {dino: (2120, 3, 224, 224); siglip: (2120, 3, 224, 224)}
        
        

        # == formulating labels ==
        input_labels = inputids.detach().clone() # create labels


        start_idx += 1 # to align with the memory padded with init memory

        cast_type = torch.float16
        if self.config.OPENVLN.flash_atten:
            cast_type = torch.bfloat16

        with torch.cuda.amp.autocast(dtype=cast_type):
            modelout = self.vlm(input_ids=inputids, attention_mask=attention_mask,pixel_values=transformed_images_tensor, labels = input_labels, img_ori_shape = (B,T), sample_valid_len = observations['lengths'], full_his = transformed_his_tensor, sample_start = start_idx,vocab_size=len(self.tokenlizer), inf_weights = collected_data['weights'])
        

        # # change if finished !
        # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        #     with torch.no_grad():
        #         modelout = self.vlm(input_ids=inputids, attention_mask=attention_mask,pixel_values=transformed_images_tensor, labels = input_labels, img_ori_shape = (B,T), sample_valid_len = observations['lengths'], full_his = transformed_his_tensor, sample_start = start_idx)
        

        pred = modelout.logits.argmax(dim=-1)
        predicted_token_id_list = pred.cpu().tolist() # (bs,token_len)

        bs_result = []
        for i in range(len(predicted_token_id_list)):
            decoded_tokens = self.tokenlizer.convert_ids_to_tokens(predicted_token_id_list[i])
            bs_result.append(decoded_tokens[-2])

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

        # self.intergrated_his_embedding = nn.Embedding(1,4096)
        # self.histor_embeddings = self.intergrated_his_embedding.weight # referencing copy, this will also be updated while loading pre-trained weights
        # self.history_intergration_attention = FFWRelativeCrossAttentionModule(4096,2,1)


        # initialize memory bank
        self.menmory_embedding = nn.Embedding(128,4096)
        self.M_init = self.menmory_embedding.weight # referencing copy, this will also be updated while loading pre-trained weights

        
        self.memory_fuser_attention = FFWRelativeCrossAttentionModule(4096,2,1)
        self.pe_layer = PositionalEncoding(4096,0.2)






    def get_input(self, input_ids, img_features, input_mask, labels):
        
        
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
            labels[:,:-1] = IGNORE_INDEX
            projectoed_patch_labels = torch.full(
                    (img_features.shape[0], img_features.shape[1]),
                    IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )

            multimodal_labels = torch.cat(
                    [
                        labels[:,:1],
                        projectoed_patch_labels,
                        labels[:,1:],
                    ],
                    dim=1,
            )


        return multimodal_embeddings, multimodal_attention_mask, multimodal_labels
        
    
    def inference_action(self,input_ids, pixel_values, full_his, img_ori_shape,sample_valid_len,sample_start):

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

        projected_cls_embeddings = self.extract_cls(full_his_patches) # (bs*T,4,dim)
        projected_his_embeddings = self.projector(projected_cls_embeddings)

        compressed_memory = self.compress_memories(projected_his_embeddings,img_ori_shape,multimodal_embeddings,sample_start)


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

    # def extract_cls(self,projected_patch_embeddings):

        
    #     expanded_histories = self.histor_embeddings.unsqueeze(0).expand(projected_patch_embeddings.shape[0], -1, -1)

    #     # intergrate histories [bsz, num_patches, llm_embed_dim] -> [bsz, 1, llm_embed_dim]
    #     integrated_his,_ = self.history_intergration_attention(query=expanded_histories.transpose(0, 1),
    #         value=projected_patch_embeddings.transpose(0, 1),
    #         query_pos=None,
    #         value_pos=None,
    #         diff_ts=None,pad_mask=None)
    #     integrated_his = integrated_his[-1].transpose(0,1) # (bs*T,1,d)
    #     projected_cls_embeddings = integrated_his

    #     return projected_cls_embeddings

    # def compress_memories(self,projected_cls_embeddings,img_ori_shape,multimodal_embeddings,sample_start):

    #     if not projected_cls_embeddings.is_contiguous():
    #         projected_cls_embeddings = projected_cls_embeddings.contiguous() # (bs*T,1,dim)
            
    #     # # format encoded histories
    #     # # (bs*T,1,dim) -> (bs,T,dim)
    #     # projected_cls_embeddings_with_T = projected_cls_embeddings.view(img_ori_shape[0],img_ori_shape[1],projected_cls_embeddings.shape[1],projected_cls_embeddings.shape[2]).squeeze(2) 
    #     # cls_embeeding_kv = projected_cls_embeddings_with_T.unsqueeze(1).expand(-1, projected_cls_embeddings_with_T.shape[1], -1, -1)
    #     # cls_embeeding_kv = cls_embeeding_kv.reshape(-1,cls_embeeding_kv.shape[2],cls_embeeding_kv.shape[3]) # (bs*T,T,dim)
    #     # his_pos = self.pe_layer(cls_embeeding_kv)

    #     # format encoded histories
    #     # (full_bs*T,1,dim) -> (token_bs,full_bs*T,dim)
    #     projected_cls_embeddings_with_T = projected_cls_embeddings.squeeze(1)  # (full_bs*T,dim)
    #     cls_embeeding_kv = projected_cls_embeddings_with_T.unsqueeze(0).expand(multimodal_embeddings.shape[0], -1, -1) # (token_bs, full_bs*T, dim)
    #     his_pos = self.pe_layer(cls_embeeding_kv)


    #     # format init memory
    #     # resulting memory in (token_bs,C,d)
    #     expanded_memory = self.M_init.unsqueeze(0).expand(cls_embeeding_kv.shape[0],-1,-1)


    #     if sample_start == -1: # inferencing, no masking
    #         # compressing
    #         compressed_memory,_ = self.memory_fuser_attention(query=expanded_memory.transpose(0, 1),
    #             value=his_pos.transpose(0, 1),
    #             query_pos=None,
    #             value_pos=his_pos,
    #             diff_ts=None,pad_mask=None,print_info=False)
    #         compressed_memory = compressed_memory[-1].transpose(0,1) # (bs*T,C,d)
    #     else:
    #         # format masking
    #         # memory masking (token_bs,his_T)
    #         token_bs = his_pos.shape[0]
    #         his_T = his_pos.shape[1]

    #         mask_all = torch.zeros(token_bs,his_T,dtype=torch.bool).to(expanded_memory.device) # set mask to be all not masking
    #         mask_effective = torch.triu(torch.ones(token_bs,token_bs,dtype=torch.bool)).to(expanded_memory.device)
    #         mask_all[:, sample_start:sample_start+token_bs] = mask_effective # set corresponding idx to be trius
    #         if sample_start+token_bs < mask_all.shape[1]:
    #             mask_all[:, sample_start+token_bs:] = True # set future idx to be false if appliable (len remains after effective len >= 1)

    #         # compressing
    #         compressed_memory,_ = self.memory_fuser_attention(query=expanded_memory.transpose(0, 1),
    #             value=his_pos.transpose(0, 1),
    #             query_pos=None,
    #             value_pos=his_pos,
    #             diff_ts=None,pad_mask=mask_all,print_info=False)
    #         compressed_memory = compressed_memory[-1].transpose(0,1) # (bs*T,C,d)

    #     print(self.M_init,self.histor_embeddings)      

    #     return compressed_memory


    def extract_cls(self,projected_patch_embeddings): # grid pooling, following Navid.

        if not projected_patch_embeddings.is_contiguous():
            projected_patch_embeddings = projected_patch_embeddings.contiguous()
        
        His_len,Token_len,C = projected_patch_embeddings.shape
        _grid = projected_patch_embeddings.view(His_len,16,16,C)
        _grid = _grid.permute(0,3,1,2)
        pool = nn.AdaptiveAvgPool2d((2,2))
        pooled = pool(_grid)

        result = pooled.permute(0,2,3,1).reshape(His_len,4,C)

        return result


    def compress_memories(self,projected_cls_embeddings,img_ori_shape,multimodal_embeddings,sample_start): # grid pooled version

        if not projected_cls_embeddings.is_contiguous():
            projected_cls_embeddings = projected_cls_embeddings.contiguous() # (his_len,4,dim)
        if not self.M_init.is_contiguous():
            self.M_init = self.M_init.contiguous() # (his_len,4,dim)

        

        token_bs = multimodal_embeddings.shape[0]
        his_T = projected_cls_embeddings.shape[0]

        projected_cls_embeddings = projected_cls_embeddings.view(-1,projected_cls_embeddings.shape[2]) # (his_len * 4,dim)
        projected_cls_embeddings = projected_cls_embeddings.unsqueeze(0).expand(token_bs, -1, -1)  # (bs,his_len * 4,dim)
     
        his_pos = self.pe_layer(projected_cls_embeddings)


        # format init memory
        # resulting memory in (token_bs,C,d)
        expanded_memory = self.M_init.unsqueeze(0).expand(token_bs,-1,-1)


        if sample_start == -1: # inferencing, no masking
            # compressing
            compressed_memory,_ = self.memory_fuser_attention(query=expanded_memory.transpose(0, 1),
                value=his_pos.transpose(0, 1),
                query_pos=None,
                value_pos=his_pos,
                diff_ts=None,pad_mask=None,print_info=False)
            compressed_memory = compressed_memory[-1].transpose(0,1) # (bs*T,C,d)
        else:
            # format masking
            # memory masking (token_bs,his_T)
            mask_all = torch.zeros(token_bs,his_T,dtype=torch.bool).to(expanded_memory.device) # set mask to be all not masking
            mask_effective = torch.triu(torch.ones(token_bs,token_bs,dtype=torch.bool)).to(expanded_memory.device)
            mask_all[:, sample_start:sample_start+token_bs] = mask_effective # set corresponding idx to be trius
            if sample_start+token_bs < mask_all.shape[1]:
                mask_all[:, sample_start+token_bs:] = True # set future idx to be false if appliable (len remains after effective len >= 1)


            # repeat 4 times on his_T dimension for grid pooled version
            mask_all = mask_all.repeat_interleave(4,dim=1)

        
            # compressing
            compressed_memory,_ = self.memory_fuser_attention(query=expanded_memory.transpose(0, 1),
                value=his_pos.transpose(0, 1),
                query_pos=None,
                value_pos=his_pos,
                diff_ts=None,pad_mask=mask_all,print_info=False)
            compressed_memory = compressed_memory[-1].transpose(0,1) # (bs*T,C,d)

        

        # print(self.M_init,"mem",compressed_memory)



    
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
        sample_start = None,
        vocab_size = None,
        inf_weights = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLN, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)

        if inference:
            return self.inference_action(input_ids, pixel_values, full_his, img_ori_shape,sample_valid_len,sample_start)


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


        


        # ==== INPUT & LABEL & MASK ====
        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_embeddings, multimodal_attention_mask, multimodal_labels = self.get_input(input_ids=input_ids,img_features=projected_patch_embeddings, input_mask=attention_mask, labels=labels)

    

        # ==== Update Memories ====
        # express histories in [cls] way
        full_his_patches = self.vision_backbone({k: full_his[k] for k in full_his}) # (bs*T,vit_token_len,dim)
        projected_cls_embeddings = self.extract_cls(full_his_patches) # (bs*T,4,dim)
        projected_his_embeddings = self.projector(projected_cls_embeddings)
        
        compressed_memory = self.compress_memories(projected_his_embeddings,img_ori_shape,multimodal_embeddings,sample_start)
        
        
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

        print(f"Forward Print {shift_logits[:,-1,-6:]}")

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # loss = fixed_cross_entropy(shift_logits, shift_labels, None, -100)

        loss = nn.functional.cross_entropy(shift_logits,shift_labels,ignore_index=-100,reduction="none")

        # loss weighting
        bs,seq = multimodal_labels[..., 1:].size()
        loss = loss.contiguous()
        inf_weights = inf_weights.contiguous()
        
        loss = loss.view(bs,seq)
        inf_weights = inf_weights.transpose(0,1) 
        weighted_loss = loss * inf_weights
        final_loss = weighted_loss.sum() / bs

        
      
        out.loss = final_loss
            
        return out





