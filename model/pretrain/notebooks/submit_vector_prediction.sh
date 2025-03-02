#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sdl-inference
#SBATCH --array=0-9
#SBATCH --mem=24G
#SBATCH -p t4v2
#SBATCH --gres=gpu:1
#SBATCH --mail-user=subercui@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# . /etc/profile.d/lmod.sh
bash ~/.bashrc
module unload cuda-12.1
module load cuda-11.7
export CUDA_HOME=/pkgs/cuda-11.7
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

data_path="/scratch/ssd004/datasets/cellxgene/3d_molecule_data/220k-lib/lmdb" # replace to your data path
# date_time=$(date "+%Y%m%d-%H%M")

weight_name="save_4CR_0509-24_0.82_0508-1056"
# weight_path="/fs01/home/haotian/SDL-LNP/model/unimol/notebooks/save_4CR_0508-24_0.85_raw_weight_custom_data/checkpoint_best.pt"
# weight_path="/fs01/home/haotian/SDL-LNP/model/unimol/notebooks/save_4CR_0508-24_0.82_0502-1907/checkpoint_best.pt"
weight_path="/fs01/home/haotian/SDL-LNP/model/unimol/notebooks/${weight_name}/checkpoint_best.pt"

results_path="/scratch/ssd004/datasets/cellxgene/unimol_pred/220k-${weight_name}/${SLURM_ARRAY_TASK_ID}" # replace to your save path
mkdir -p results_path

cd /fs01/home/haotian/SDL-LNP/model/unimol/notebooks/

dict_name='dict.txt'
task_name="${SLURM_ARRAY_TASK_ID}"
task_num=1
loss_func='finetune_mse'
batch_size=64
only_polar=0 # -1 all h; 0 no h
conf_size=11
seed=0
metric="valid_agg_rmse"

python ../unimol/infer.py --user-dir ../unimol $data_path --task-name $task_name --valid-subset test \
       --results-path $results_path \
       --num-workers 4 --ddp-backend=c10d --batch-size $batch_size \
       --task mol_finetune --loss $loss_func --arch unimol_base \
       --classification-head-name $task_name --num-classes $task_num \
       --dict-name $dict_name --conf-size $conf_size \
       --only-polar $only_polar \
       --path $weight_path \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple \
       --fixed-validation-seed 1
