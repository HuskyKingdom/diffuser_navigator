#!/bin/bash -l

# job name
#SBATCH -J atd_single_old

# output file name
#SBATCH -o atd_single_old.out

# running time
#SBATCH -t 72:00:00  

# partition
#SBATCH -p gpu

#SBATCH -N 1 
#SBATCH --gres=gpu:1

# CPU Core
#SBATCH -n 6 

# mail notice
#SBATCH --mail-user=sgyson10@liverpool.ac.uk
#SBATCH --mail-type=ALL 

# load modules
module purge
module load apps/cmake/3.25.1/gcc-11.2.0


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
conda activate vlnce

export NCCL_DEBUG=INFO

# job run
export HF_HOME=./ && python run.py   --exp-config diffuser_baselines/config/d3diffuser_old.yaml   --run-type train
