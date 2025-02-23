import gc
import os
import random
import psutil
import warnings
import zlib
from collections import defaultdict
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from gym import Space
from habitat import Config
import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

# from habitat_baselines.common.environments import get_env_class
from habitat.core.environments import get_env_class

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from diffuser_baselines.models.utils import batch_obs
# from habitat_baselines.utils.common import batch_obs

import contextlib
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import construct_envs,construct_envs_process
from vlnce_baselines.common.utils import extract_instruction_tokens

import torch.nn.functional as Fuc
import torch.nn as nn

# fsdp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from diffuser_baselines.common.base_openvln_trainer import BaseVLNCETrainer
from torch.cuda.amp import autocast, GradScaler


from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    requeue_job,
)

# from habitat_baselines.rl.ddppo.algo.ddp_utils import (
#     EXIT,
#     REQUEUE,
#     add_signal_handlers,
#     init_distrib_slurm,
#     load_interrupted_state,
#     requeue_job,
#     save_interrupted_state,
# )

from torch.utils.data import DataLoader, DistributedSampler    

import torch.nn.functional as Fuc

from contextlib import contextmanager


# openvln
import wandb
from peft import get_peft_model, LoraConfig
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm



@contextmanager
def maybe_tqdm(total=None, desc=None, leave=True, dynamic_ncols=True):
    if dist.get_rank() == 0:
        yield tqdm.tqdm(total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)
    else:
        # 提供一个简单的占位符
        yield contextlib.nullcontext()

@contextmanager
def maybe_trange(start_epoch, end_epoch, *args, **kwargs):
    """
    A wrapper to yield `tqdm.trange` if on rank 0, otherwise yield a plain range.

    Args:
        start_epoch (int): The starting epoch for the range.
        end_epoch (int): The ending epoch for the range.
        *args: Additional arguments to pass to `tqdm.trange` or `range`.
        **kwargs: Additional keyword arguments to pass to `tqdm.trange` or `range`.

    Yields:
        tqdm.trange or range: Progress indicator for rank 0 or plain range for other ranks.
    """
    if dist.get_rank() == 0:
        yield tqdm.trange(start_epoch, end_epoch, *args, **kwargs)
    else:
        yield range(start_epoch, end_epoch, *args)

@contextmanager
def maybe_tqdm_iterable(iterable, *args, **kwargs):
    if dist.get_rank() == 0:
        yield tqdm.tqdm(iterable, *args, **kwargs)
    else:
        yield iterable


def compress_data(data):
    return zlib.compress(msgpack_numpy.packb(data, use_bin_type=True))




    

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
        # 'rgb_features': [],          
        # 'depth_features': [],         
        'rgb': [],
        'gt_actions': [],            
        'trajectories':[],
        'prev_actions':[],
        'padding_mask': [],
        'lengths': [],
        'weights': [],
        'ins_text': [],
        'progress': [],
        'labels': [],
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
        # rgb_feat = torch.tensor(sample_dict['rgb_features'])  # (len_seq, 2048, 4, 4)
        # depth_feat = torch.tensor(sample_dict['depth_features'])  # (len_seq, 128, 4, 4)

        rgb = torch.tensor(sample_dict['rgb']) # (len_seq, 224, 224, 3)
     
        gt_actions = torch.tensor(sample[2])  # (len_seq)
        trajectories = torch.tensor(sample[3])  # (len_seq, 4)
        prev_actions = torch.tensor(sample[1]) 
        progress = torch.tensor(sample_dict['progress'])

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
        # pad_rgb_feat = _pad_helper(rgb_feat, max_len)
        # pad_depth_feat = _pad_helper(depth_feat, max_len)
        pad_rgb = _pad_helper(rgb, max_len)
        pad_gt_actions = _pad_helper(gt_actions, max_len)
        pad_prev_actions = _pad_helper(prev_actions, max_len)
        pad_trajectories = _pad_helper(trajectories, max_len)
        pad_weights = _pad_helper(weights, max_len)
        pad_progress = _pad_helper(progress,max_len)



        # Create padding_mask for this sample
        mask = torch.ones(max_len, dtype=torch.bool)
        mask[:seq_len] = False  # False represents real data

        # Append padded data to collected_data
        collected_data['instruction'].append(instr)

        # collected_data['rgb_features'].append(pad_rgb_feat)
        # collected_data['depth_features'].append(pad_depth_feat)

        collected_data['rgb'].append(pad_rgb)


        collected_data['gt_actions'].append(pad_gt_actions)
        collected_data["prev_actions"].append(pad_prev_actions)
        collected_data['trajectories'].append(pad_trajectories)
        collected_data['padding_mask'].append(mask) # padding mask for dec_input
        collected_data["weights"].append(pad_weights),
        collected_data["progress"].append(pad_progress)


    # Stack each list in collected_data into a tensor
    for key in collected_data:
        if key == 'lengths' or key == "ins_text" or key == "labels":
            continue
        collected_data[key] = torch.stack(collected_data[key], dim=0)

    # pipeline test only
    collected_data["labels"] = [
    "Short text",
    "This is a slightly longer text",
    "This is an even longer text that is used to test cases where text lengths vary",]

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

                
                trajectory = msgpack_numpy.unpackb(zlib.decompress(data), raw=False)
                # trajectory = msgpack_numpy.unpackb(data, raw=False)
    
                return trajectory



class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
        rank=-1,
        world_size=-1
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 1
        self._preload = []
        self._preload_index = []
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.num_loaded_data = 0

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

                    load_index = self.load_ordering.pop()
                    preload_data = msgpack_numpy.unpackb(
                        zlib.decompress(txn.get(str(load_index).encode())), raw=False
                    )
                    if 'ep_id' in preload_data[0].keys():
                        del preload_data[0]['ep_id']
                    new_preload.append(preload_data)

                    lengths.append(len(new_preload[-1][-1]))
                    self._preload_index.append(load_index)

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in sorted_ordering:
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    # def __next__(self):
    #     obs, prev_actions, oracle_actions = self._load_next()

    #     for k, v in obs.items():
    #         obs[k] = torch.from_numpy(np.copy(v))

    #     prev_actions = torch.from_numpy(np.copy(prev_actions))
    #     oracle_actions = torch.from_numpy(np.copy(oracle_actions))

      

    #     return (obs, prev_actions, oracle_actions)

    def __next__(self):
        obs, prev_actions, oracle_actions, trajectories, ins_text = self._load_next()


        for k, v in obs.items():
            obs[k] = np.copy(v)

        prev_actions = np.copy(prev_actions)
        oracle_actions = np.copy(oracle_actions)
        trajectories = np.copy(trajectories)
        ins_text = np.copy(ins_text)

        # Return data
        return (obs, prev_actions, oracle_actions, trajectories, ins_text)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            per_proc = int(np.floor(self.length / self.world_size))

            start = per_proc * self.rank
            end = min(start + per_proc, self.length)
            assert end - start == per_proc 
        else:
            per_proc = int(np.floor(self.length / self.world_size))
            per_worker = int(np.floor(per_proc / worker_info.num_workers))

            start = per_worker * worker_info.id + per_proc * self.rank
            end = min(start + per_worker, self.length)
            assert end - start == per_worker 

        self.load_ordering = list(
            reversed(_block_shuffle(list(range(start, end)), self.preload_size))
        )
        if self.rank == 0:
            logger.info(f'[Dataset]: Total data size {self.length}')

        return self
    
    def __len__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return int(np.floor(self.length / self.world_size))
        else:
            per_proc = int(np.floor(self.length / self.world_size))
            return int(np.floor(per_proc / worker_info.num_workers)) * worker_info.num_workers




@baseline_registry.register_trainer(name="openvln_trainer_fsdp")
class OpenVLNTrainerFSDP(BaseVLNCETrainer):
    def __init__(self, config=None):
        self.lmdb_features_dir = config.IL.DAGGER.lmdb_features_dir.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )
        self.envs = None
        super().__init__(config)

        # init wandb
        wandb.init(

            project="yhscode-university-of-liverpool",
            # track hyperparameters and run metadata
            config={
            "learning_rate": config.OPENVLN.LR,
            "epochs": config.IL.epochs,
            "bs": config.IL.batch_size,
            }
        )

        self.scaler = GradScaler()

        

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def _update_dataset(self, data_it):

        # init ----------------
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        gc.collect()


        if self.envs is None:
            allocated_cuda_memory = torch.cuda.memory_allocated(device=self.local_rank) / (1024 * 1024 * 1024)
            if allocated_cuda_memory > 6:
                self.config.defrost()
                self.config.NUM_PROCESSES = int((12 - allocated_cuda_memory) // 2.5)
                self.config.freeze()
                logger.info("cuda memory is not enough, processes reduce to ", int((12 - allocated_cuda_memory) // 2.5))
            self.config.defrost()
            self.config.TASK_CONFIG.DATASET.split_num = self.world_size
            self.config.TASK_CONFIG.DATASET.split_rank = self.local_rank
            self.config.freeze()
            self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        else:
            self.envs.resume_all()

        # envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid

        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES,
            1,
            device=self.device,
            dtype=torch.long,
        )

        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, dtype=torch.uint8, device=self.device
        )

        # env reset ----------------
        observations = self.envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        episodes = [[] for _ in range(self.envs.num_envs)]
        skips = [False for _ in range(self.envs.num_envs)]
        # Populate dones with False initially
        dones = [False for _ in range(self.envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        p = self.config.IL.DAGGER.p
        # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
        beta = 0.0 if p == 0.0 else p ** data_it

        ensure_unique_episodes = beta == 1.0

        # building hooks ----------------

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
            ep_ids_collected = {
                ep.episode_id for ep in self.envs.current_episodes()
            }


        def writeCache(cache):
            with ThreadPool(8) as workers:
                cache = workers.map(compress_data, cache)
            txn = lmdb_env.begin(write=True)
            existed_size = lmdb_env.stat()["entries"]
            for i, v in enumerate(cache):
                txn.put(str(existed_size + i).encode(), v)
            txn.commit()

        dist.barrier()
        time.sleep(1*self.local_rank)
        cache = []

        prev_instructions = ["None" for i in range(self.envs.num_envs)]


        # start iteration ----------------

        with lmdb.open(self.lmdb_features_dir,map_size=int(self.config.IL.DAGGER.lmdb_map_size),) as lmdb_env, torch.no_grad():
            
            required_size = (data_it+1) * self.config.IL.DAGGER.update_size
            remain_update_size = required_size - lmdb_env.stat()["entries"]
            start_id = lmdb_env.stat()["entries"]

            if self.local_rank == 0:
                pbar = tqdm.tqdm(total=remain_update_size, smoothing=0.01, desc=f"Collecting Data, DAgger iter {data_it}")


            # start iterations ----------------
                
            # txn = lmdb_env.begin(write=True)

            while collected_eps < remain_update_size and not (lmdb_env.stat()["entries"] >= required_size):
                print(f"rank {self.local_rank} ; collected eps {collected_eps} ; stats {lmdb_env.stat()['entries']}; required {required_size}")
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = self.envs.current_episodes()


                # collect logic ----------------

                for i in range(self.envs.num_envs):

                    if not dones[i]: # store instruction if not done
                        prev_instructions[i] = self.envs.current_episodes()[i].instruction.instruction_text

                    if dones[i] and not skips[i]:

                        # if len(episodes[i]) > 250:
                        #     episodes[i] = []
                        #     self.policy.module.clear_his()
                        #     prev_actions = torch.zeros(
                        #         self.config.NUM_PROCESSES,
                        #         1,
                        #         device=self.device,
                        #         dtype=torch.long,
                        #     )
                        #     continue

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
                        

                        # modif
                        prev_instructions[i] += "If you deviate from the correct path or do not see the clues above, try to explore and get back on track."
                    

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                            np.array([step[3] for step in ep], dtype=np.float32),
                            np.array([prev_instructions[i]])
                        ]

                        # put ----------------
                        
                        #print(f"prev {np.array([step[1] for step in ep], dtype=np.int64)} | oracle {np.array([step[2] for step in ep], dtype=np.int64)}")
                        # txn.put(
                        #     str(start_id + collected_eps).encode(),
                        #     msgpack_numpy.packb(
                        #         transposed_ep, use_bin_type=True
                        #     ),
                        # )
                        # pbar.update()
                        # collected_eps += 1

                        cache.append(transposed_ep)
                        if self.local_rank == 0:
                            pbar.update(lmdb_env.stat()["entries"] - start_id - pbar.n)
                        collected_eps += 1

                        # commit ----------------
                        ava_mem = float(psutil.virtual_memory().available) / 1024 / 1024 / 1024
                        if (len(cache) % self.config.IL.DAGGER.lmdb_commit_frequency == 0 or ava_mem < 10) and len(cache) != 0 and not lmdb_env.stat()["entries"] >= required_size:
                            writeCache(cache)
                            del cache
                            cache = []
                            print(f"rank {self.local_rank} ; WRITE! ; stats {lmdb_env.stat()['entries']}")


                        # if (
                        #     collected_eps
                        #     % self.config.IL.DAGGER.lmdb_commit_frequency
                        # ) == 0:
                        #     txn.commit()
                        #     txn = lmdb_env.begin(write=True)

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
                        # reset
                        self.policy.module.clear_his()
                        prev_actions = torch.zeros(
                                self.config.NUM_PROCESSES,
                                1,
                                device=self.device,
                                dtype=torch.long,
                            )
    

                if ensure_unique_episodes:
                    (
                        self.envs,
                        not_done_masks,
                        batch,
                        _,
                    ) = self._pause_envs(
                        envs_to_pause,
                        self.envs,
                        not_done_masks,
                        batch,
                    )
                    if self.envs.num_envs == 0:
                        break
                
                # act ---------------- modif
                    
                ins_text = []
                for i in range(self.envs.num_envs):
                    ins_text.append(self.envs.current_episodes()[i].instruction.instruction_text + "If you deviate from the correct path or do not see the clues above, try to explore and get back on track.")   

          

                if (torch.rand_like(prev_actions.long(), dtype=torch.float) < beta):
                    # action from expert
                    actions = self.policy.module.act(
                        batch, prev_actions, encode_only=True, ins_text=ins_text
                    )
                    actions = batch[expert_uuid].long()
                else:
                    # action from model
                    actions = self.policy.module.act(
                        batch, prev_actions, encode_only=False, ins_text=ins_text
                    )
                                

                # random_values = torch.rand_like(actions, dtype=torch.float)
                # mask = random_values < beta
                # actions = torch.where(mask, batch[expert_uuid].long(), actions)
                

                # collect inference features ----------------

                for i in range(self.envs.num_envs):
                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"]

                    if depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]


                    pos = self.envs.call_at(i, "get_state", {"observations": {}})

                    
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
                prev_actions.copy_(actions)

                # step action ----------------

                outputs = self.envs.step([a[0].item() for a in actions])
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
                
            print(f"rank {self.local_rank} finished collecting, with imdb size {lmdb_env.stat()['entries']}")
        
        self.envs.close()
        self.envs = None

        if rgb_hook is not None:
            rgb_hook.remove()
        if depth_hook is not None:
            depth_hook.remove()
        
        # reset
        self.policy.module.clear_his()

    def train(self) -> None:
        
        if not dist.is_initialized():
            self.local_rank, tcp_store = init_distrib_slurm("NCCL")
            print("NCCL INITIALIZED.")
        else:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            tcp_store = torch.distributed.distributed_c10d._get_default_store()

        torch.cuda.set_device(self.local_rank)
        add_signal_handlers()
    
        self.world_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
    
        if self.world_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
    

        """Main method for training DAgger."""
        if self.config.IL.DAGGER.preload_lmdb_features:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True, lock=False)
            except lmdb.Error as err:
                logger.error(
                    "Cannot open database for teacher forcing preload."
                )
                raise err
        else:
            print("OpenVLN training pipeline does not support Dagger collecting. Expect IL.DAGGER.preload_lmdb_features as True.")
            

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


        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            4, 
        )


        with (TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) if self.world_rank == 0

            else contextlib.suppress() ) as writer:


            for dagger_it in range(self.config.IL.DAGGER.iterations):

                # get dataset ---
                step_id = self.step_id

                # dist.barrier()
                # if torch.cuda.is_available():
                #     with torch.cuda.device(self.device):
                #         torch.cuda.empty_cache()
                # gc.collect()


                pre_collected_dataset = IWTrajectoryDataset(
                    self.lmdb_features_dir,
                    lmdb_map_size=self.config.IL.DAGGER.lmdb_map_size,
                    batch_size=self.config.IL.batch_size,
                    rank=self.local_rank,
                    world_size=self.world_size
                )
                diter = torch.utils.data.DataLoader(
                    pre_collected_dataset,
                    batch_size=self.config.IL.batch_size // self.world_size, 
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=4,
                    sampler=None
                )



                    
                if torch.cuda.is_available():
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()


                epoch_loss = 0
                num_epoch_batch = 0

                with maybe_trange(self.start_epoch, self.config.IL.epochs, desc="Epochs", dynamic_ncols=True) as epoch_range:

                    for epoch in epoch_range:

                        # diter.sampler.set_epoch(epoch)

                        with maybe_tqdm_iterable(
                            diter,
                            total=pre_collected_dataset.length // pre_collected_dataset.batch_size,
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
                                if k != "ins_text" and k != "labels" else v
                                )
                                for k, v in batch.items()
                                }



                                loss = self._update_agent(
                                    batch
                                )

                                wandb.log({"loss": loss})
                                epoch_loss += loss

                                if self.world_rank == 0: #ddp
                                    writer.add_scalar(
                                        f"train_loss_iter_{dagger_it}_{self.config.JobName}", loss, step_id
                                    )
                                step_id += 1  # noqa: SIM113
                                num_epoch_batch += 1

                        if self.world_rank == 0: # fsdp save 
                            epoch_loss /= num_epoch_batch
                            logger.info(f"epoch loss: {epoch_loss}  | steps {num_epoch_batch} | On Diffuser iter {dagger_it}, Epoch {epoch}.")
                            wandb.log({"epoch loss": epoch_loss, "steps": num_epoch_batch, "Epoch": epoch})
                            epoch_loss = 0
                            num_epoch_batch = 0

                        # fsdp save logic
                        if (dagger_it * self.config.IL.epochs + epoch + 1) % self.config.OPENVLN.saving_frequency == 0:
                            self.save_checkpoint(
                                f"ckpt.{dagger_it * self.config.IL.epochs + epoch + 1}.pth"
                            )
                        else:
                            print("Not to save.")
                            

                        dist.barrier() #ddp
        
        dist.destroy_process_group() #ddp
        wandb.finish()
                    

    def grad_clipping(self):  # @save

        self.policy.vlm.clip_grad_norm_(max_norm=1.0)
        

    def _update_agent(
        self,
        observations,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
        config = None,
    ):
        

        loss = self.policy.build_loss(observations)  # Access the underlying module

        

        
        with torch.no_grad():
            loss_tensor = loss.clone()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor = loss_tensor / self.world_size


        
        self.scaler.scale(loss).backward()

        self.grad_clipping()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()


        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
            gc.collect() 


        return loss_tensor.item()


    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        num_actions: int,
    ) -> None:


        torch.cuda.empty_cache()
        
        train = True
        
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy(
            config,
        )

        self.policy.to(self.device)


        trainable_params = [param for param in self.policy.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.config.OPENVLN.LR
        )


        if config.lr_Schedule: # train 250 + 500 + 750  + 1000 + 1250 + 1500 + 1750 + 2000 + 2250 + 2500 
            if not config.dagger: # self.config.IL.DAGGER.update_size // self.config.IL.batch_size
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config.OPENVLN.LR, pct_start=0.35, 
                                                steps_per_epoch=7862, epochs=self.config.IL.epochs)
            else:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.config.OPENVLN.LR, pct_start=0.35, 
                                                total_steps = self.config.IL.DAGGER.update_size * 55 // self.config.IL.batch_size * self.config.IL.epochs)


        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")

            self.policy.vlm.load_state_dict(ckpt_dict["state_dict"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")


        # fsdp
        self.reduce_in_full_precision = False
        reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
        fsdp_precision_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=reduce_buffer_dtype, buffer_dtype=reduce_buffer_dtype)

        if config.OPENVLN.stage not in {"full-finetune", "vla-full-train", "vla-sandwich-train"}:
            self.policy.vlm.vision_backbone.to(dtype=self.policy.vlm.vision_backbone.half_precision_dtype)
        

        if train:
            self.policy.vlm = FSDP(
            self.policy.vlm,
            auto_wrap_policy=self.policy.vlm.get_fsdp_wrapping_policy(),
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
            )
            

            

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")


        # MEM LEFT 25GB
        


        

        

    def save_checkpoint(self, file_name: str) -> None: # rewrite for ddp
        """Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint
        """


        with FSDP.state_dict_type(self.policy.vlm, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):

            full_vlm_state_dict = self.policy.vlm.state_dict()

            checkpoint = {
                "state_dict": full_vlm_state_dict,
                "optim_state": None,
                "scheduler_state": None,
                "config": self.config,
            }

            if self.world_rank == 0:
                torch.save(
                checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
                )



            # model_state_dicts = {
            #     mkey: OrderedDict() for mkey in (self.policy.vlm.all_module_keys)
            # }
            # # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
            # for key, param in full_vlm_state_dict.items():
            #     for mkey in model_state_dicts:
            #         if key.startswith(mprefix := f"{mkey}."):
            #             model_state_dicts[mkey][key.removeprefix(mprefix)] = param

            # checkpoint_path = file_name

            # # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
            # torch.save({"model": model_state_dicts}, checkpoint_path)
