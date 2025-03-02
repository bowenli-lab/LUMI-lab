#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=unimol-pretrain
#SBATCH --mem=220GB
#SBATCH --gres=gpu:4
#SBATCH --partition=a100
#SBATCH --qos=a100_bowang
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=subercui@gmail.com
#SBATCH --mail-type=ALL

# log the sbatch environment
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
export NCCL_IB_DISABLE=1

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

data_path=/datasets/cellxgene/3d_molecule_data/ligands/
date_time=$(date "+%Y%m%d-%H%M")
save_dir=/fs01/datasets/cellxgene/3d_molecule_save/pretrain-$date_time/
n_gpu=$SLURM_GPUS_ON_NODE
MASTER_PORT=10086
lr=1e-4
wd=1e-4
batch_size=64
update_freq=1
heads=8
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000

# copyt this file to save_dir
mkdir -p $save_dir
cp $0 $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
# torchrun --nproc_per_node=$n_gpu $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ../unimol --train-subset train --valid-subset valid \
    --num-workers 8 --ddp-backend=c10d \
    --task unimol --loss unimol --arch unimol_base \
    --encoder-attention-heads $heads --dropout 0.0 \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
    --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
    --update-freq $update_freq --seed $seed \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
    --max-update $max_steps --log-interval 10 --log-format simple \
    --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
    --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
    --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
    --save-dir $save_dir --only-polar $only_polar --mode train
