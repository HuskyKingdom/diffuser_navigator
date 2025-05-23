#!/bin/bash -l

# job name
#SBATCH -J openvln

# output file name
#SBATCH -o out/openvln.out

# running time
#SBATCH -t 2:00:00  

# partition
#SBATCH -p gpulowsmall

#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=gpu13

# CPU Core
#SBATCH -n 30

# mail notice
#SBATCH --mail-user=sgyson10@liverpool.ac.uk
#SBATCH --mail-type=ALL 

# load modules
# module purge
# module load apps/cmake/3.25.1/gcc-11.2.0
# module load libs/nvidia-cuda/12.1.1/bin

module purge
module load apps/cmake/3.25.1/gcc-11.2.0
module load libs/nvidia-cuda/12.1.1/bin


# conda
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/users/sgyson10/volatile/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/users/sgyson10/volatile/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/users/sgyson10/volatile/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/users/sgyson10/volatile/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# activate conda
conda activate openvln
export PYTHONDONTWRITEBYTECODE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:24
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=1

# job run
export HF_HOME=./ && CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 run.py   --exp-config diffuser_baselines/config/openvln.yaml   --run-type train
