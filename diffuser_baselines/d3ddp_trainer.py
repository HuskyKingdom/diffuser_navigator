import gc
import os
import random
import warnings
from multiprocessing import Pool
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
import contextlib
from vlnce_baselines.common.aux_losses import AuxLosses
from diffuser_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens

# ddp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from torch.utils.data import DataLoader, DistributedSampler    


import torch.nn.functional as Fuc

# ddpm pgb management

from contextlib import contextmanager

def compress_data(data):
    """Compress the data for efficient LMDB storage."""
    return zlib.compress(msgpack_numpy.packb(data, use_bin_type=True))

@contextmanager
def maybe_tqdm(total=None, desc=None, leave=True, dynamic_ncols=True):
    if dist.get_rank() == 0:
        yield tqdm.tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)
    else:
        # 提供一个简单的占位符
        yield contextlib.nullcontext()

@contextmanager
def maybe_trange(*args, **kwargs):
    if dist.get_rank() == 0:
        yield tqdm.trange(*args, **kwargs)
    else:
        yield range(*args)

@contextmanager
def maybe_tqdm_iterable(iterable, *args, **kwargs):
    if dist.get_rank() == 0:
        yield tqdm.tqdm(iterable, *args, **kwargs)
    else:
        yield iterable



class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self
    

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
        trajectories: (len_seq,4),
        ins_text; str
    ]
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - len(t)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)
    

    collected_data = {
        'instruction': [],           
        'rgb_features': [],          
        'depth_features': [],         
        'gt_actions': [],            
        'trajectories':[],
        'prev_actions':[],
        'padding_mask': [],
        'lengths': [],
        'weights': [],
        'ins_text': [],
    }
    
    # Transpose the batch to separate each component
    transposed = list(zip(*batch))

    # Compute max sequence length in the batch
    lengths = [len(sample[0]['instruction']) for sample in batch]
    collected_data['lengths'] = torch.tensor(lengths).long()
    max_len = max(lengths)

    for sample in batch:

        # Extract data from the sample
        sample_dict = sample[0]
        instr = torch.tensor(sample_dict['instruction'][0])  # (len_seq, 200) only take one instruction
        rgb_feat = torch.tensor(sample_dict['rgb_features'])  # (len_seq, 2048, 4, 4)
        depth_feat = torch.tensor(sample_dict['depth_features'])  # (len_seq, 128, 4, 4)
        gt_actions = torch.tensor(sample[2])  # (len_seq)
        trajectories = torch.tensor(sample[3])  # (len_seq, 4)
        prev_actions = torch.tensor(sample[1]) 

        collected_data["ins_text"].append(sample[4][0]) # instruction text

        # compute weights
        inflection_weights = torch.tensor([1.0, 3.2])
        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (gt_actions[1:] != gt_actions[:-1]).long(),
            ]
        )
        weights = inflection_weights[inflections]

        seq_len = gt_actions.shape[0]

        # Pad sequences to the maximum length
        pad_rgb_feat = _pad_helper(rgb_feat, max_len)
        pad_depth_feat = _pad_helper(depth_feat, max_len)
        pad_gt_actions = _pad_helper(gt_actions, max_len)
        pad_prev_actions = _pad_helper(prev_actions, max_len)
        pad_trajectories = _pad_helper(trajectories, max_len)
        pad_weights = _pad_helper(weights, max_len)



        # Create padding_mask for this sample
        mask = torch.ones(max_len, dtype=torch.bool)
        mask[:seq_len] = False  # False represents real data

        # Append padded data to collected_data
        collected_data['instruction'].append(instr)
        collected_data['rgb_features'].append(pad_rgb_feat)
        collected_data['depth_features'].append(pad_depth_feat)
        collected_data['gt_actions'].append(pad_gt_actions)
        collected_data["prev_actions"].append(pad_prev_actions)
        collected_data['trajectories'].append(pad_trajectories)
        collected_data['padding_mask'].append(mask) # padding mask for dec_input
        collected_data["weights"].append(pad_weights)

    # Stack each list in collected_data into a tensor
    for key in collected_data:
        if key == 'lengths' or key == "ins_text":
            continue
        collected_data[key] = torch.stack(collected_data[key], dim=0)



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


@baseline_registry.register_trainer(name="ddp_d3diffuser")
class DiffuserTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        super().__init__(config)
        self.envs = None

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()


    def _update_dataset(self, data_it):

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        # --------- 并行化逻辑开始（参考代码B）-----------
        # 根据本进程的rank和world_size进行数据分片
        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.split_num = self.world_size
        self.config.TASK_CONFIG.DATASET.split_rank = self.local_rank
        self.config.freeze()
        
        # 构造并行环境
        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        # --------- 并行化逻辑结束 -----------

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
        dones = [False for _ in range(envs.num_envs)]

        p = self.config.IL.DAGGER.p
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
            rgb_hook = self.policy.module.navigator.rgb_encoder.cnn.register_forward_hook(
                hook_builder(rgb_features)
            )

        depth_features = None
        depth_hook = None
        if not self.config.MODEL.DEPTH_ENCODER.trainable:
            depth_features = torch.zeros((1,), device="cpu")
            depth_hook = self.policy.module.navigator.depth_encoder.visual_encoder.register_forward_hook(
                hook_builder(depth_features)
            )

        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = {ep.episode_id for ep in envs.current_episodes()}

        prev_instructions = ["None" for i in range(envs.num_envs)]

        with tqdm.tqdm(
            total=self.config.IL.DAGGER.update_size, dynamic_ncols=True, disable=(self.local_rank != 0)
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            txn = lmdb_env.begin(write=True)

            while collected_eps < self.config.IL.DAGGER.update_size:
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if not dones[i]:
                        prev_instructions[i] = envs.current_episodes()[i].instruction.instruction_text

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
                            np.array([prev_instructions[i]])
                        ]
                        
                        txn.put(
                            str(start_id + collected_eps).encode(),
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ),
                        )

                        if self.local_rank == 0:
                            pbar.update()
                        collected_eps += 1

                        if (
                            collected_eps
                            % self.config.IL.DAGGER.lmdb_commit_frequency
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)

                        if ensure_unique_episodes:
                            if current_episodes[i].episode_id in ep_ids_collected:
                                envs_to_pause.append(i)
                            else:
                                ep_ids_collected.add(
                                    current_episodes[i].episode_id
                                )

                    if dones[i]:
                        episodes[i] = []
                        self.policy.module.clear_his()

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

                if (torch.rand_like(prev_actions.long(), dtype=torch.float) < beta):
                    # action from expert
                    actions = self.policy.module.act(
                        batch,prev_actions,encode_only = True
                    )
                    actions = batch[expert_uuid].long()
                else:
                    ins_text = envs.current_episodes()[0].instruction.instruction_text
                    # action from model
                    actions = self.policy.module.act(
                        batch,prev_actions,encode_only = False,ins_text=ins_text
                    )

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
                actions = torch.where(skips, torch.zeros_like(actions), actions)
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)
                prev_actions.copy_(actions)

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

        self.policy.module.clear_his()

        # --------- 并行化逻辑：结束时Barrier同步，确保所有进程都完成数据更新 -----------
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
        # --------- 并行化逻辑结束 -----------

        logger.info("Dataset collection complete.")



    def train(self) -> None:

        # ddp
        self.local_rank, tcp_store = init_distrib_slurm(
        "NCCL"  # Ensure this path is correct
        )
        add_signal_handlers()
    
        self.world_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
    
        if self.world_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
    



        """Main method for training DAgger."""
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True, lock=False) # ddp
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.config.IL.DAGGER.lmdb_map_size), lock=False # ddp
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())

        EPS = self.config.IL.DAGGER.expert_policy_sensor
        if EPS not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(EPS)

        self.config.defrost()
        
        # ddp
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_IDS = [self.local_rank]
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_ENVIRONMENTS
        )
        self.device = (
            torch.device("cuda", self.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )



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
            train=True
        )


        with (TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) if self.world_rank == 0
            else contextlib.suppress() ) as writer:
            for diffuser_it in range(self.config.IL.DAGGER.iterations):

                # get dataset ---
                step_id = 0
                if not self.config.IL.DAGGER.preload_lmdb_features:
                    self._update_dataset(
                        diffuser_it + (1 if self.config.IL.load_from_ckpt else 0)
                    )
                torch.distributed.barrier()
                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                # get dataset ---
                    
                diffusion_dataset = TrajectoryDataset(self.lmdb_features_dir,self.config.IL.DAGGER.lmdb_map_size,self.config.IL.batch_size)

                # ddp
                ddp_sampler = DistributedSampler(diffusion_dataset)
                diter = torch.utils.data.DataLoader(
                    diffusion_dataset,
                    batch_size=self.config.IL.batch_size,
                    shuffle=False,
                    sampler=ddp_sampler,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=4,
                )

                    
                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()


                epoch_loss = 0
                num_epoch_batch = 0

                with maybe_trange(self.config.IL.epochs, desc="Epochs", dynamic_ncols=True) as epoch_range:
                    for epoch in epoch_range:
                        with maybe_tqdm_iterable(
                            diter,
                            total=diffusion_dataset.length // diffusion_dataset.batch_size,
                            leave=False,
                            dynamic_ncols=True,
                            desc="Batches"
                        ) as batch_iter:
                            
                            for batch in batch_iter:
                            
                                batch = {
                            k: (
                                v.to(
                                    device=self.device,
                                    dtype=torch.float32,
                                    non_blocking=True,
                                )
                                if k != "ins_text" else v
                            )
                            for k, v in batch.items()
                        }

                                loss = self._update_agent(
                                    batch
                                )

                                epoch_loss += loss

                                if self.world_rank == 0: #ddp
                                    writer.add_scalar(
                                        f"train_loss_iter_{diffuser_it}_{self.config.JobName}", loss, step_id
                                    )
                                step_id += 1  # noqa: SIM113
                                num_epoch_batch += 1

                            if self.world_rank == 0: #ddp
                                if (diffuser_it * self.config.IL.epochs + epoch) % 200 == 0:
                                    self.save_checkpoint(
                                        f"ckpt.{diffuser_it * self.config.IL.epochs + epoch}.pth"
                                    )
                                else:
                                    print(diffuser_it * self.config.IL.epochs + epoch, "Not to save.")

                                epoch_loss /= num_epoch_batch
                                epoch_loss = 0
                                num_epoch_batch = 0
                                logger.info(f"epoch loss: {loss}  | On Diffuser iter {diffuser_it}, Epoch {epoch}.")

                        dist.barrier() #ddp
        
        dist.destroy_process_group() #ddp




    def _update_agent( # ddp
        self,
        observations,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        loss = self.policy.module.build_loss(observations)  # Access the underlying module
        
        with torch.no_grad():
            loss_tensor = loss.clone()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor = loss_tensor / self.world_size
        
        loss = loss / loss_accumulation_scalar
        loss.backward()
        
        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.config.lr_Schedule:
                self.scheduler.step()
        
        return loss_tensor.item()
    



    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        num_actions: int,
        train = False,
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
            self.policy.parameters(), lr=self.config.DIFFUSER.LR
        )


        if config.lr_Schedule: # train 250 + 500 + 750  + 1000 + 1250 + 1500 + 1750 + 2000 + 2250 + 2500 
            if not config.dagger:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config.DIFFUSER.LR, pct_start=0.35, 
                                                steps_per_epoch=540 // self.world_size, epochs=self.config.IL.epochs)
            else:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config.DIFFUSER.LR, pct_start=0.35, 
                                                total_steps=55000 // self.world_size)


        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            
            # load policy from ddp
            state_dict = ckpt_dict['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            ckpt_dict['state_dict'] = new_state_dict

            self.policy.load_state_dict(ckpt_dict["state_dict"])

            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        # ddp
        if train:
            self.policy = DDP(self.policy, device_ids=[self.local_rank], output_device=self.local_rank)
        
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")
