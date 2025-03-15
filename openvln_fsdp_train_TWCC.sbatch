#!/bin/bash -l

# job name
#SBATCH -J openvln_fsdp

# output file name
#SBATCH -o out/openvln_fsdp.out

# running time
#SBATCH -t 48:00:00  

# partition
#SBATCH -p gp4d

#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
// #SBATCH --cpus-per-task=4 


// # cpu core / nodes
#SBATCH -n 48

# mail notice
#SBATCH --mail-user=sgyson10@liverpool.ac.uk
#SBATCH --mail-type=ALL 


module purge
module load cuda/12.3
module load gcc10/10.2.1

# conda
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/work/u5966600/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/u5966600/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work/u5966600/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/u5966600/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# activate conda
conda activate openvln

export PYTHONDONTWRITEBYTECODE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TOKENIZERS_PARALLELISM=true
# export NCCL_P2P_DISABLE=1

# multi-node
export NODES=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_ADDR=$(scontrol show node $NODES | grep -oP 'NodeAddr=\K\S+')
export MASTER_PORT=29500


echo $MASTER_ADDR 
echo $SLURM_NODEID

# net
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=1

# job run
export HF_HOME=./

srun torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  --rdzv_id $SLURM_JOB_ID \
  run.py --exp-config diffuser_baselines/config/openvln_fsdp_TWCC.yaml --run-type train

