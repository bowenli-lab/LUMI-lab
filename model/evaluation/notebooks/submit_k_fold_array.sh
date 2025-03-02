#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=array-finetune
#SBATCH --gres=gpu:1
#SBATCH --qos=m4
#SBATCH --mail-user=pangkuantony@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=kfold_log/%x-%j.out
#SBATCH --error=kfold_log/%x-%j.err
#SBATCH --array=1-10
module unload python

conda activate unimol
source /h/pangkuan/miniconda3/envs/unimol/bin/activate

export HF_DATASETS_CACHE="/scratch/ssd004/scratch/pangkuan/hf_cache"

cd /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/

run_arg=$(sed -n "${SLURM_ARRAY_TASK_ID}p" case_list)

# split by space
weight_name=$(echo $run_arg | cut -d' ' -f1)
fold_num=$(echo $run_arg | cut -d' ' -f2)

echo "weight_name: ${weight_name}"
echo "fold_num: ${fold_num}"


ckpt_path="/checkpoint/${USER}/${SLURM_JOB_ID}"

/h/pangkuan/miniconda3/envs/unimol/bin/python /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/unimol_huggingface_kfold.py --weight-name ${weight_name} --loss-type "normalized_preference_log" --parallel-fold ${fold_num} --ckpt-path ${ckpt_path}

/h/pangkuan/miniconda3/envs/unimol/bin/python /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/unimol_huggingface_kfold.py --weight-name ${weight_name} --loss-type "mse" --parallel-fold ${fold_num} --ckpt-path ${ckpt_path}

/h/pangkuan/miniconda3/envs/unimol/bin/python /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/unimol_huggingface_kfold.py --weight-name ${weight_name} --loss-type "preference_log" --parallel-fold ${fold_num} --ckpt-path ${ckpt_path}