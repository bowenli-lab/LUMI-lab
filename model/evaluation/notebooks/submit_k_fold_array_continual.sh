#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=continual-array-sweep
#SBATCH --gres=gpu:1
#SBATCH --qos=m4
#SBATCH --mail-user=pangkuantony@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=sweep_log/%x-%j.out
#SBATCH --error=sweep_log/%x-%j.err
#SBATCH --array=0-119
module unload python

conda activate unimol
source /h/pangkuan/miniconda3/envs/unimol/bin/activate

export HF_DATASETS_CACHE="/scratch/ssd004/scratch/pangkuan/hf_cache"

cd /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/

# Define the learning rates to sweep over
lr_values=(1e-5 5e-5 2e-5)

# Define the contrastive_loss values to sweep over
contrastive_loss_values=(10.0 20.0)

# Define the ligand_weight values to sweep over
ligand_weight_values=(0.2 0.4)

# Define the lipid_weight values to sweep over
lipid_weight_values=(0.5 1.0)

# Define the number of folds
num_folds=5

# Calculate the total number of combinations
num_lr_values=${#lr_values[@]}
num_contrastive_loss_values=${#contrastive_loss_values[@]}
num_ligand_weight_values=${#ligand_weight_values[@]}
num_lipid_weight_values=${#lipid_weight_values[@]}
total_combinations=$((num_lr_values * num_contrastive_loss_values * num_ligand_weight_values * num_lipid_weight_values * num_folds))

# Calculate indices
fold_index=$(($SLURM_ARRAY_TASK_ID % num_folds))
lipid_weight_index=$((($SLURM_ARRAY_TASK_ID / num_folds) % num_lipid_weight_values))
ligand_weight_index=$((($SLURM_ARRAY_TASK_ID / (num_folds * num_lipid_weight_values)) % num_ligand_weight_values))
contrastive_loss_index=$((($SLURM_ARRAY_TASK_ID / (num_ligand_weight_values * num_lipid_weight_values * num_folds)) % num_contrastive_loss_values))
lr_index=$(($SLURM_ARRAY_TASK_ID / (num_contrastive_loss_values * num_ligand_weight_values * num_lipid_weight_values * num_folds)))

# Get the values for this job
lr=${lr_values[$lr_index]}
contrastive_loss=${contrastive_loss_values[$contrastive_loss_index]}
ligand_weight=${ligand_weight_values[$ligand_weight_index]}
lipid_weight=${lipid_weight_values[$lipid_weight_index]}
fold_number=$((fold_index + 1))

echo "lr=$lr, contrastive_loss=$contrastive_loss, ligand_weight=$ligand_weight, lipid_weight=$lipid_weight, fold=$fold_number, in total $total_combinations combinations"


weight_name=continual-pretrain-a100_sweep-lr${lr}-cw${contrastive_loss}-lg${ligand_weight}-lp${lipid_weight}

ckpt_path="/checkpoint/${USER}/${SLURM_JOB_ID}"

/h/pangkuan/miniconda3/envs/unimol/bin/python /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/unimol_huggingface_kfold.py --weight-name ${weight_name} --loss-type "normalized_preference_log" --parallel-fold ${fold_number} --ckpt-path ${ckpt_path}

/h/pangkuan/miniconda3/envs/unimol/bin/python /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/unimol_huggingface_kfold.py --weight-name ${weight_name} --loss-type "mse" --parallel-fold ${fold_number} --ckpt-path ${ckpt_path}

/h/pangkuan/miniconda3/envs/unimol/bin/python /h/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/unimol_huggingface_kfold.py --weight-name ${weight_name} --loss-type "preference_log" --parallel-fold ${fold_number} --ckpt-path ${ckpt_path}