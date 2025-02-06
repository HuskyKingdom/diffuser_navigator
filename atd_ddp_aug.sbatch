#!/bin/bash -l

# job name
#SBATCH -J atd_ddp_aug

# output file name
#SBATCH -o out/atd_ddp_aug.out

# running time
#SBATCH -t 2:00:00  

# partition
#SBATCH -p dev

#SBATCH -N 1 
#SBATCH --gres=gpu:4

# CPU Core
#SBATCH -n 24

# mail notice
#SBATCH --mail-user=sgyson10@liverpool.ac.uk
#SBATCH --mail-type=ALL 

# load modules
module purge
module cuda/12.2


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


# activate conda
conda activate openvln

export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=true
# export NCCL_P2P_DISABLE=1


# job run sbatch --account=MST113264 atd_ddp_aug.sbatch
export HF_HOME=./ && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run.py   --exp-config diffuser_baselines/config/d3ddp_aug.yaml   --run-type train
