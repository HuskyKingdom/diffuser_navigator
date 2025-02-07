#!/bin/bash -l

# job name
#SBATCH -J openvln

# output file name
#SBATCH -o out/openvln.out

# running time
#SBATCH -t 2:00:00  

# partition
#SBATCH -p normal

#SBATCH -N 1
#SBATCH --gres=gpu:8

# CPU Core
#SBATCH -n 48 

# mail notice
#SBATCH --mail-user=sgyson10@liverpool.ac.uk
#SBATCH --mail-type=ALL 

# load modules
# module purge
# module load apps/cmake/3.25.1/gcc-11.2.0
# module load libs/nvidia-cuda/12.1.1/bin

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
export NCCL_P2P_DISABLE=1

# job run
export HF_HOME=./ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 run.py   --exp-config diffuser_baselines/config/openvln.yaml   --run-type train
