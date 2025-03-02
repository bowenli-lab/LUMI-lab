#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000
#SBATCH --mail-user=subercui@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=kfold_log/%x-%j.out
#SBATCH --error=kfold_log/%x-%j.err

bash ~/.bashrc
module unload cuda-12.1
module load cuda-12.1
export CUDA_HOME=/pkgs/cuda-12.1
nvcc --version

# >>> conda initialize >>>
__conda_setup="$('/pkgs/anaconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/pkgs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/pkgs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/pkgs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate ~/.conda/envs/unimol
which python

export HF_DATASETS_CACHE="/datasets/cellxgene/hf_cache"

weight_name="continual-pretrain-a100_large_epoch-lr2e-5-cw10.0-lg0.5.new"

cd ~/SDL-LNP/model/evaluation/notebooks/
python unimol_huggingface_kfold.py \
    --weight-name ${weight_name} \
    --loss-type "weighted_mse" \
    --lmdb-data "exp1102_export.lmdb" \
    --save-model
