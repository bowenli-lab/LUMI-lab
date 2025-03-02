#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --mem=24G
#SBATCH --qos=m5
#SBATCH -p t4v2
#SBATCH --gres=gpu:1
#SBATCH --mail-user=pangkuantony@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=pred_log/%x-%j.out
#SBATCH --error=pred_log/%x-%j.err
module unload python

conda activate unimol
source /h/pangkuan/miniconda3/envs/unimol/bin/activate

# SLURM_ARRAY_TASK_ID=0

data_path="/scratch/ssd004/datasets/cellxgene/3d_molecule_data/220k-lib/lmdb/"  # replace to your data path
results_path="/scratch/ssd004/datasets/cellxgene/unimol_pred/pred1920/${SLURM_ARRAY_TASK_ID}"  # replace to your save path
MASTER_PORT=10086
n_gpu=1
dict_name='dict.txt'
weight_path="/scratch/ssd004/datasets/cellxgene/3d_molecule_save/weights/checkpoint_best.pt"
task_name="${SLURM_ARRAY_TASK_ID}"
task_num=1
loss_func='finetune_mse'
batch_size=64
only_polar=0 # -1 all h; 0 no h
conf_size=11



# cp "/h/pangkuan/dev/SDL-LNP/model/unimol/notebooks/dict.txt" ${data_path}
/h/pangkuan/miniconda3/envs/unimol/bin/python ../unimol/infer.py --user-dir ../unimol $data_path --task-name $task_name --valid-subset test \
       --results-path $results_path \
       --num-workers 4 --ddp-backend=c10d --batch-size $batch_size \
       --task mol_finetune --loss $loss_func --arch unimol_base \
       --classification-head-name $task_name --num-classes $task_num \
       --dict-name $dict_name --conf-size $conf_size \
       --only-polar $only_polar  \
       --path $weight_path  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple  \
       --fixed-validation-seed 1