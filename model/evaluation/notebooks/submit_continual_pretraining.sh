#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000
#SBATCH --mail-user=pangkuantony@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=pred_log/%x-%j.out
#SBATCH --error=pred_log/%x-%j.err
module unload python

conda activate unimol
source /h/pangkuan/miniconda3/envs/unimol/bin/activate

export HF_DATASETS_CACHE="/scratch/ssd004/scratch/pangkuan/hf_cache"

cd /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/
/h/pangkuan/miniconda3/envs/unimol/bin/python unimol_huggingface_continual_pretraining.py