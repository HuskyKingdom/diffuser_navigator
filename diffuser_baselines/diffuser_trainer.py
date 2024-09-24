import gc
import os
import random
import warnings
from collections import defaultdict
from gym import Space
from habitat import Config
import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from diffuser_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens

import torch.nn.functional as Fuc

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


# make input into batched tensors & -1 pad on the oracle actions
# def collate_fn(batch):    
#     """
#     [
#     {instruction:(len_seq,200); progress:(len_seq,1); rgb_features:(len_seq,2048,4,4); depth_features:(len_seq,128,4,4)},
#     prev_actions:(len_seq),
#     gt_actions:(len_seq),
#     ]
#     """

#     # num of feature timestep prediction

    

#     F = 2 # action_lenth - 1

#     collected_data = {
#         'instruction': [],
#         'rgb_features': [],
#         'depth_features': [],
#         'gt_actions': []
#     }
    
#     for sample in batch:
#         len_seq = sample[0]['instruction'].shape[0]
        
#         # randomly sample timestep t in the range [0, len_seq-1]
#         t = random.randint(0, len_seq - 1)
        
#         # Handle instruction, rgb_features, depth_features
#         collected_data['instruction'].append(torch.tensor(sample[0]['instruction'][t]))
#         collected_data['rgb_features'].append(torch.tensor(sample[0]['rgb_features'][t]))
#         collected_data['depth_features'].append(torch.tensor(sample[0]['depth_features'][t]))
        
#         # Handle gt_actions by selecting from t to t+F, padding with STOP(0) if out of bounds
#         if t + F < len_seq:
#             gt_action_segment = sample[2][t:t+F+1]
#         else:
#             gt_action_segment = sample[2][t:] 
#             padding_size = (t + F + 1) - len_seq 
#             gt_action_segment = np.concatenate([gt_action_segment, np.full(padding_size, 0)]) # padding with STOP action if exceed

#         collected_data['gt_actions'].append(torch.tensor(gt_action_segment))
    
#     # Stack into batched tensors
#     collected_data['instruction'] = torch.stack(collected_data['instruction'], dim=0)
#     collected_data['rgb_features'] = torch.stack(collected_data['rgb_features'], dim=0)
#     collected_data['depth_features'] = torch.stack(collected_data['depth_features'], dim=0)
#     collected_data['gt_actions'] = torch.stack(collected_data['gt_actions'], dim=0)


#     return collected_data
    


def collate_fn(batch):
    """
    batch 是样本列表
    每个样本是：
    [
        {
            'instruction': (len_seq, 200),
            'progress': (len_seq, 1),
            'rgb_features': (len_seq, 2048, 4, 4),
            'depth_features': (len_seq, 128, 4, 4)
        },
        prev_actions: (len_seq),
        gt_actions: (len_seq),
        trajectories: (len_seq,4)
    ]
    """

    F = 2  # 预测的未来动作数量 - 1

    collected_data = {
        'instruction': [],            # 时间 t 的指令
        'rgb_features': [],           # 时间 t 的 RGB 特征
        'depth_features': [],         # 时间 t 的深度特征
        'gt_actions': [],             # 从 t 到 t+F 的专家动作
        'seq_timesteps': [],            # 历史序列的长度（t）
        'trajectories':[],
        'proprioceptions': []  # agent pose t -1
    }

    t_list = []

    # 首先确定每个样本的 t
    for sample in batch:
        len_seq = sample[0]['instruction'].shape[0]
        t = random.randint(0, len_seq - 1)
        t_list.append(t)


    for idx, sample in enumerate(batch):
        len_seq = sample[0]['instruction'].shape[0]
        t = t_list[idx]

        # 时间 t 的当前观测
        collected_data['instruction'].append(torch.tensor(sample[0]['instruction'][t]))
        collected_data['rgb_features'].append(torch.tensor(sample[0]['rgb_features'][t]))
        collected_data['depth_features'].append(torch.tensor(sample[0]['depth_features'][t]))

        # 从 t 到 t+F 的专家动作，必要时用 STOP(0) 填充
        if t + F < len_seq:
            gt_action_segment = sample[2][t:t+F+1]
        else:
            gt_action_segment = sample[2][t:]
            padding_size = (t + F + 1) - len_seq
            gt_action_segment = np.concatenate([gt_action_segment, np.full(padding_size, 0)])  # 用 STOP 动作填充

        # 从 t+1 到 t+1+F 的trajectory，必要时用 STOP(0) 填充
        if t+1 + F < len_seq:
            gt_traj = sample[3][t+1:t+1+F+1]
        else:
            last_valid_traj = sample[3][-1]
            padding_size = (t + F + 2) - len_seq
            gt_traj = sample[3][t+1:]
            padding_traj = np.tile(last_valid_traj, (padding_size, 1))
            gt_traj = np.concatenate([gt_traj, padding_traj])
            # gt_traj = np.concatenate([gt_traj, np.full((padding_size,4), 0.0)])  # 用 zero 动作填充



        collected_data['gt_actions'].append(torch.tensor(gt_action_segment))
        collected_data['trajectories'].append(torch.tensor(gt_traj))


        # pose at t
        collected_data['proprioceptions'].append(sample[3][t])

        # 记录历史序列长度（t）
        collected_data['seq_timesteps'].append(t)


    collected_data['proprioceptions'] = np.array(collected_data['proprioceptions'])


    # 将收集的数据堆叠成批量张量
    collected_data['instruction'] = torch.stack(collected_data['instruction'], dim=0)
    collected_data['rgb_features'] = torch.stack(collected_data['rgb_features'], dim=0)
    collected_data['depth_features'] = torch.stack(collected_data['depth_features'], dim=0)
    collected_data['gt_actions'] = torch.stack(collected_data['gt_actions'], dim=0)
    collected_data['seq_timesteps'] = torch.tensor(collected_data['seq_timesteps'])
    collected_data['trajectories'] = torch.stack(collected_data['trajectories'], dim=0)
    collected_data['proprioceptions'] = torch.tensor(collected_data['proprioceptions'])

    
    
    return collected_data









def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]



class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_features_dir, map_size, batch_size):
        """
        trajectories: list of episodes, where each episode is a list of timesteps.
        max_timestep: maximum number of timesteps to consider in each episode.
        """
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.batch_size = batch_size
        self.map_size = map_size

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):

        with lmdb.open(
        self.lmdb_features_dir,
        map_size=self.map_size,
        readonly=True,
        lock=False,
        ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:

                data = txn.get(str(index).encode())
                
                if data is None:
                    raise IndexError(f"Index {index} out of range in database")

                
                trajectory = msgpack_numpy.unpackb(data, raw=False)
    
                return trajectory



class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.load_ordering.pop()).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


@baseline_registry.register_trainer(name="diffuser")
class DiffuserTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _update_dataset(self, data_it):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid

        prev_actions = torch.zeros(
            envs.num_envs,
            1,
            device=self.device,
            dtype=torch.long,
        )

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        episodes = [[] for _ in range(envs.num_envs)]
        skips = [False for _ in range(envs.num_envs)]
        # Populate dones with False initially
        dones = [False for _ in range(envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        p = self.config.IL.DAGGER.p
        # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
        beta = 0.0 if p == 0.0 else p ** data_it

        ensure_unique_episodes = beta == 1.0

        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())

            return hook

        rgb_features = None
        rgb_hook = None
        if not self.config.MODEL.RGB_ENCODER.trainable:
            rgb_features = torch.zeros((1,), device="cpu")
            rgb_hook = self.policy.navigator.rgb_encoder.cnn.register_forward_hook(
                hook_builder(rgb_features)
            )

        depth_features = None
        depth_hook = None
        if not self.config.MODEL.DEPTH_ENCODER.trainable:
            depth_features = torch.zeros((1,), device="cpu")
            depth_hook = self.policy.navigator.depth_encoder.visual_encoder.register_forward_hook(
                hook_builder(depth_features)
            )

        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = {
                ep.episode_id for ep in envs.current_episodes()
            }

        with tqdm.tqdm(
            total=self.config.IL.DAGGER.update_size, dynamic_ncols=True
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

            while collected_eps < self.config.IL.DAGGER.update_size:
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if dones[i] and not skips[i]:
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        )
                        del traj_obs[expert_uuid]
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()
                            if self.config.IL.DAGGER.lmdb_fp16:
                                traj_obs[k] = traj_obs[k].astype(np.float16)

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                            np.array([step[3] for step in ep], dtype=np.float32),
                        ]
                        txn.put(
                            str(start_id + collected_eps).encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
                        )

                        pbar.update()
                        collected_eps += 1

                        if (
                            collected_eps
                            % self.config.IL.DAGGER.lmdb_commit_frequency
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                        if ensure_unique_episodes:
                            if (
                                current_episodes[i].episode_id
                                in ep_ids_collected
                            ):
                                envs_to_pause.append(i)
                            else:
                                ep_ids_collected.add(
                                    current_episodes[i].episode_id
                                )

                    if dones[i]:
                        episodes[i] = []

                if ensure_unique_episodes:
                    (
                        envs,
                        not_done_masks,
                        batch,
                        _,
                    ) = self._pause_envs(
                        envs_to_pause,
                        envs,
                        not_done_masks,
                        batch,
                    )
                    if envs.num_envs == 0:
                        break

                actions = self.policy.act(
                    batch,[1],encode_only = True
                ) # inference for getting features only

                actions = batch[expert_uuid].long() # oracle actions only

                for i in range(envs.num_envs):
                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"]

                    if depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]


                    pos = envs.call_at(i, "get_state", {"observations": {}})

                    
                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch[expert_uuid][i].item(),
                            pos
                        )
                    )

                skips = batch[expert_uuid].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)

                outputs = envs.step([a[0].item() for a in actions])
                observations, _, dones, _ = [list(x) for x in zip(*outputs)]

                

                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                
                

                    
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

            txn.commit()

        envs.close()
        envs = None

        if rgb_hook is not None:
            rgb_hook.remove()
        if depth_hook is not None:
            depth_hook.remove()

    def train(self) -> None:
        """Main method for training DAgger."""
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.IL.DAGGER.lmdb_map_size),
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        EPS = self.config.IL.DAGGER.expert_policy_sensor
        if EPS not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(EPS)

        self.config.defrost()

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.IL.DAGGER.p == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        observation_space, action_space = self._get_spaces(self.config)

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            4, 
        )

        # self.policy.eval()
        
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            for diffuser_it in range(self.config.IL.DAGGER.iterations):

                # get dataset ---
                step_id = 0
                if not self.config.IL.DAGGER.preload_lmdb_features:
                    self._update_dataset(
                        diffuser_it + (1 if self.config.IL.load_from_ckpt else 0)
                    )
                    assert 1==2
                # get dataset ---
                    
                diffusion_dataset = TrajectoryDataset(self.lmdb_features_dir,self.config.IL.DAGGER.lmdb_map_size,self.config.IL.batch_size)

                diter = torch.utils.data.DataLoader(
                    diffusion_dataset,
                    batch_size=self.config.IL.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=1,
                )

                    
                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()


                epoch_loss = 0
                num_epoch_batch = 0
                for epoch in tqdm.trange(
                    self.config.IL.epochs, dynamic_ncols=True
                ):
                    for batch in tqdm.tqdm(
                        diter,
                        total=diffusion_dataset.length // diffusion_dataset.batch_size,
                        leave=False,
                        dynamic_ncols=True,
                    ):
             

                        batch = {
                            k: v.to(
                                device=self.device,
                                dtype=torch.float32,
                                non_blocking=True,
                            )
                            for k, v in batch.items()
                        }

                        loss = self._update_agent(
                            batch
                        )

                        epoch_loss += loss
            
                        writer.add_scalar(
                            f"train_loss_iter_{diffuser_it}", loss, step_id
                        )
                        step_id += 1  # noqa: SIM113
                        num_epoch_batch += 1

                    self.save_checkpoint(
                        f"ckpt.{diffuser_it * self.config.IL.epochs + epoch}.pth"
                    )

                    epoch_loss /= num_epoch_batch
                    epoch_loss = 0
                    num_epoch_batch = 0
                    logger.info(f"epoch loss: {loss} | Batches processed: {step_id}. | On Diffuser iter {diffuser_it}, Epoch {epoch}.")


    def _update_agent(
        self,
        observations,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        

        loss = self.policy.build_loss(
            observations
        )


        loss = loss / loss_accumulation_scalar
        loss.backward()

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()



        return loss.item()


    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        num_actions: int,
    ) -> None:
        
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy(
            config,
            num_actions=num_actions,
            embedding_dim = config.DIFFUSER.embedding_dim,
            num_attention_heads= config.DIFFUSER.num_attention_heads,
            num_layers = config.DIFFUSER.num_layers,
            diffusion_timesteps = config.DIFFUSER.diffusion_timesteps
        )
        self.policy.to(self.device)

        self.optimizer = torch.optim.AdamW(
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