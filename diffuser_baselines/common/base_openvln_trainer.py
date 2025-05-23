import json
import os
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import gc
import jsonlines
import torch
import torch.nn.functional as F
import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry

# from habitat_baselines.common.environments import get_env_class
from habitat.core.environments import get_env_class

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

# from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.rl.ddppo.ddp_utils import is_slurm_batch_job

# from habitat_baselines.utils.common import batch_obs
from diffuser_baselines.models.utils import batch_obs

from habitat_extensions.utils import generate_video, observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens
from transformers import BertTokenizer

class BaseVLNCETrainer(BaseILTrainer):
    """A base trainer for VLN-CE imitation learning."""

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:


        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.IL.lr
        )
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")




    def _get_spaces(
        self, config: Config, envs: Optional[Any] = None
    ) -> Tuple[Space]:
        """Gets both the observation space and action space.

        Args:
            config (Config): The config specifies the observation transforms.
            envs (Any, optional): An existing Environment. If None, an
                environment is created using the config.

        Returns:
            observation space, action space
        """
        if envs is not None:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]

        else:
            env = get_env_class(self.config.ENV_NAME)(config=config)
            observation_space = env.observation_space
            action_space = env.action_space

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        return observation_space, action_space
    
    def save_checkpoint(self, file_name: str) -> None:
        """Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint
        """
        checkpoint = {
            "state_dict": self.policy.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def save_checkpoint_for_old(self, file_name: str) -> None:
        """Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint
        """
        checkpoint = {
            "state_dict": self.policy.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name),_use_new_zipfile_serialization=False
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        T, N = corrected_actions.size()

        recurrent_hidden_states = torch.zeros(
            N,
            self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        AuxLosses.clear()

        distribution = self.policy.build_distribution(
            observations, recurrent_hidden_states, prev_actions, not_done_masks
        )

        logits = distribution.logits
        logits = logits.view(T, N, -1)

        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)

        loss = action_loss + aux_loss
        loss = loss / loss_accumulation_scalar
        loss.backward()

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
        return loss.item(), action_loss.item(), aux_loss

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        batch,
        rgb_frames=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            not_done_masks = not_done_masks[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            batch,
            rgb_frames,
        )


    def action_pop(): # 
        return
    

    def _train_eval_checkpoint(
        self,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """
        logger.info(f"checkpointing...")

        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt)

        split = "val_unseen" # evaluation split

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        # init policy

        self.policy.eval()

        observations = envs.reset()
        
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )

  

        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=200) if config.use_pbar else None
        log_str = (
            f"Test"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        action_candidates = [[]]
    

        while envs.num_envs > 0 and len(stats_episodes) < 200:
            
            # retrive pose & append history rgb
            all_pose = []
            for i in range(envs.num_envs):
                pos = envs.call_at(i, "get_state", {"observations": {}})
                all_pose.append(pos)
            
            current_episodes = envs.current_episodes()

  
            actions = self.policy.act(batch,prev_actions,print_info=False)
            prev_actions.copy_(actions)

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]


            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )


            # reset envs and obserations if necessary
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)
                    

                if not dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                # reset
                self.policy.clear_his()
                prev_actions[i] = torch.zeros(1, dtype=torch.long)

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                # print(f"infos: {infos[i]}")

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                not_done_masks,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                not_done_masks,
                batch,
                rgb_frames,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )
        logger.info(f"Episodes evaluated: {num_episodes}")


        return aggregated_stats["success"]

    
    def append_text_with_weights_to_image(self, image, tokens, weights):
        import numpy as np
        import cv2
        import textwrap
        from matplotlib import cm
        import re


        weights = weights[1:-1]
       

        # print(tokens,len(tokens))
        # print(weights,len(weights))
        # assert 1==2
        # print(len(tokens),len(weights))
        assert len(tokens) == len(weights), "Tokens and weights must have the same length."
        assert all(0 <= w <= 1 for w in weights), "Weights must be between 0 and 1."

        h, w, c = image.shape
        font_size = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Create a blank canvas for the text
        blank_image = np.zeros((100, w, 3), dtype=np.uint8)

        # Map weights to colors using a colormap
        colormap = cm.get_cmap("viridis")
        colors = [tuple(int(c * 255) for c in colormap(weight)[:3]) for weight in weights]

        # Determine starting positions for the text
        x, y = 10, 30
        for token, color in zip(tokens, colors):
            # Get text size
            textsize = cv2.getTextSize(token, font, font_size, font_thickness)[0]
            # Add token to the image with the corresponding color
            cv2.putText(
                blank_image,
                token,
                (x, y),
                font,
                font_size,
                color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )
            # Move to the next token's position
            x += textsize[0] + 10  # Add some spacing

            # Break to a new line if the text exceeds image width
            if x >= w - 10:
                x = 10
                y += textsize[1] + 10

        # Adjust the final blank image height based on the last y-coordinate
        blank_image = blank_image[: y + 10, :, :]

        # Concatenate the original image with the new text image
        final = np.concatenate((image, blank_image), axis=0)
        return final


    def append_probs_to_image(self, image, labels, probs):
        """
        在原始 image 下面拼接一段文字条，
        每个 label 后面跟它对应的 prob（0~1），
        然后根据 prob 用 viridis colormap 着色。
        """
        import numpy as np
        import cv2
        from matplotlib import cm

        assert len(labels) == len(probs), "标签数和概率数必须一致"
        assert all(0.0 <= p <= 1.0 for p in probs), "所有概率必须在 [0,1]"

        h, w, _ = image.shape
        # 初始一块 100px 高的黑色画布，宽度和 image 一致
        blank_h = 100
        blank = np.zeros((blank_h, w, 3), dtype=np.uint8)

        # colormap
        cmap = cm.get_cmap("viridis")
        colors = [
            tuple(int(c * 255) for c in cmap(p)[:3])
            for p in probs
        ]

        # 文本绘制参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5
        thickness = 1
        x, y = 10, 30

        for label, prob, color in zip(labels, probs, colors):
            text = f"{label}:{prob:.2f}"
            (tx, ty), _ = cv2.getTextSize(text, font, font_size, thickness)
            cv2.putText(
                blank,
                text,
                (x, y),
                font,
                font_size,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            x += tx + 10
            # 换行
            if x >= w - 10:
                x = 10
                y += ty + 10

        # 裁剪多余空白
        blank = blank[: y + 10, :, :]
        # 垂直拼接
        return np.concatenate((image, blank), axis=0)

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """

        

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        logger.info(f"checkpoint_path: {checkpoint_path}")
        

        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt)

        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{split}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        
        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        # init policy
        self._initialize_policy(
            config,
            True,
            4, 
        )

        # envs.sim.config.sim_cfg.allow_sliding = True

        self.policy.eval()

        observations = envs.reset()
        
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )

  

        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        action_candidates = [[]]
        


        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            
            # retrive pose & append history rgb
            all_pose = []
            for i in range(envs.num_envs):
                pos = envs.call_at(i, "get_state", {"observations": {}})
                all_pose.append(pos)
            
            current_episodes = envs.current_episodes()

            # print info for anylyze
            ins_text = current_episodes[i].instruction.instruction_text
            # ins_text += "If you deviate from the correct path or do not see the clues above, try to explore and get back on track."

            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
                gc.collect() 

            with torch.no_grad():
                actions, inf_logits, attn_weights = self.policy.act(batch,prev_actions,print_info=True,ins_text=ins_text)
            prev_actions.copy_(actions)

            outputs = envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            


            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )
            
            class_labels = ["<FORWARD>", "<LEFT>", "<RIGHT>", "<STOP>"]

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    

                    class_probs = inf_logits[:,-1,-4:]
                    class_probs = torch.softmax(class_probs, dim=1)
                    class_probs = class_probs[i].cpu().tolist()
                    frame = self.append_probs_to_image(frame, class_labels, class_probs)



                    rgb_frames[i].append(frame)
                    
                  
                    

                if not dones[i]:
                    continue

                # dones 

            
                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                # reset
                prev_actions[i] = torch.zeros(1, dtype=torch.long)
                self.policy.clear_his()


                # vis attention weights
                Lq, Nk = attn_weights.shape
                attn_avg = attn_weights.reshape(Lq, Nk//4, 4).mean(axis=2)
                attn_sum = attn_avg.sum(axis=0)

                

                # import matplotlib.pyplot as plt
                # import numpy as np
                # from matplotlib.colors import Normalize

                # plt.rcParams.update({
                #     'font.size': 14,         
                #     'axes.titlesize': 16,    
                #     'axes.labelsize': 14,     
                #     'xtick.labelsize': 12,    
                #     'ytick.labelsize': 12,  
                #     'figure.titlesize': 16   
                # })

                # data = attn_avg[:, 30:]
                # vmin, vmax = np.percentile(data, (2, 98))
                # norm = Normalize(vmin=vmin, vmax=vmax)

                # plt.figure(figsize=(8, 6))
                # plt.imshow(
                #     data,
                #     aspect='auto',
                #     interpolation='nearest',
                #     origin='lower',
                #     cmap='Reds',
                #     norm=norm                    # 使用归一化
                # )
                # plt.clim(vmin, vmax)

                # plt.xlim(29, data.shape[1] - 1)
                # plt.xlabel(f'Timestep Pos (Key) [30 … {Nk-1}]')
                # plt.ylabel(f'Context Compress Vector (Query) (0 … {Lq-1})')
                # plt.title(f"Context Compress Vector Attention Distribution of Episode {ep_id}")
                # plt.colorbar(label='Attention Intensity')
                # plt.tight_layout()
                
                # plt.savefig(f"data/heatmap_{ep_id}.pdf", format='pdf', bbox_inches='tight')


                # plt.figure(figsize=(10, 4))
                # x = np.arange(attn_sum.shape[0])
                # plt.bar(x, attn_sum)
                # plt.xlabel('Timestep t')
                # plt.ylabel('Attention')
                # plt.title('Attention over Timesteps')
                # plt.tight_layout()
                # plt.savefig(f"data/_{ep_id}.pdf", format='pdf')

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

                print(f"infos: {infos[i]}")

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]

                num_done = len(stats_episodes)
                metrics = next(iter(stats_episodes.values())).keys()
                agg_stats = {
                    metric: sum(info[metric] for info in stats_episodes.values()) / num_done
                    for metric in metrics
                }
                logger.info(f"--- Cumulative stats after {num_done} episodes ---")
                for k, v in agg_stats.items():
                    logger.info(f"{k}: {v:.6f}")

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                not_done_masks,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                not_done_masks,
                batch,
                rgb_frames,
            )



        envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)

    def inference(self) -> None:
        """Runs inference on a checkpoint and saves a predictions file."""

        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.INFERENCE.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.INFERENCE.LANGUAGES
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = config.INFERENCE.CKPT_PATH
        config.TASK_CONFIG.TASK.MEASUREMENTS = []
        config.TASK_CONFIG.TASK.SENSORS = [
            s for s in config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s
        ]
        config.ENV_NAME = "VLNCEInferenceEnv"
        config.freeze()

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        observations = envs.reset()



        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        episode_predictions = defaultdict(list)

        # episode ID --> instruction ID for rxr predictions format
        instruction_ids: Dict[str, int] = {}

        # populate episode_predictions with the starting state
        current_episodes = envs.current_episodes()
        for i in range(envs.num_envs):
            episode_predictions[current_episodes[i].episode_id].append(
                envs.call_at(i, "get_info", {"observations": {}})
            )
            if config.INFERENCE.FORMAT == "rxr":
                ep_id = current_episodes[i].episode_id
                k = current_episodes[i].instruction.instruction_id
                instruction_ids[ep_id] = int(k)

        with tqdm.tqdm(
            total=sum(envs.count_episodes()),
            desc=f"[inference:{self.config.INFERENCE.SPLIT}]",
        ) as pbar:
            while envs.num_envs > 0:
                current_episodes = envs.current_episodes()
                with torch.no_grad():
                    actions, rnn_states = self.policy.act(
                        batch,
                        rnn_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=not config.INFERENCE.SAMPLE,
                    )
                    prev_actions.copy_(actions)

                outputs = envs.step([a[0].item() for a in actions])
                observations, _, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

                # reset envs and observations if necessary
                for i in range(envs.num_envs):
                    episode_predictions[current_episodes[i].episode_id].append(
                        infos[i]
                    )
                    if not dones[i]:
                        continue

                    observations[i] = envs.reset_at(i)[0]
                    prev_actions[i] = torch.zeros(1, dtype=torch.long)
                    pbar.update()

                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                envs_to_pause = []
                next_episodes = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue

                    if next_episodes[i].episode_id in episode_predictions:
                        envs_to_pause.append(i)
                    else:
                        episode_predictions[
                            next_episodes[i].episode_id
                        ].append(
                            envs.call_at(i, "get_info", {"observations": {}})
                        )
                        if config.INFERENCE.FORMAT == "rxr":
                            ep_id = next_episodes[i].episode_id
                            k = next_episodes[i].instruction.instruction_id
                            instruction_ids[ep_id] = int(k)

                (
                    envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                    _,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                )

        envs.close()

        if config.INFERENCE.FORMAT == "r2r":
            with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(episode_predictions, f, indent=2)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )
        else:  # use 'rxr' format for rxr-habitat leaderboard
            predictions_out = []

            for k, v in episode_predictions.items():

                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if path[-1] != p["position"]:
                        path.append(p["position"])

                predictions_out.append(
                    {
                        "instruction_id": instruction_ids[k],
                        "path": path,
                    }
                )

            predictions_out.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(
                config.INFERENCE.PREDICTIONS_FILE, mode="w"
            ) as writer:
                writer.write_all(predictions_out)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )
