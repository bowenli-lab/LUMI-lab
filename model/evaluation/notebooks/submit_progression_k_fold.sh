#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000
#SBATCH --mail-user=pangkuantony@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=kfold_log/%x-%j.out
#SBATCH --error=kfold_log/%x-%j.err
module unload python

conda activate unimol
source /h/pangkuan/miniconda3/envs/unimol/bin/activate

export HF_DATASETS_CACHE="/scratch/ssd004/scratch/pangkuan/hf_cache"

weight_name="10x-sdl-contrastive-ckpt-2"
kfold_mode="progression"

cd /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/
/h/pangkuan/miniconda3/envs/unimol/bin/python /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/unimol_huggingface_kfold.py --weight-name $weight_name --kfold-mode $kfold_mode