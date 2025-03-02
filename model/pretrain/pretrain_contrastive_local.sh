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

# . /etc/profile.d/lmod.sh
bash ~/.bashrc
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


conda activate ~/miniconda3/envs/unimol
which python


data_path=/home/sdl/vector_cellxgene_data/cleaned_ligands
date_time=$(date "+%Y%m%d-%H%M")
save_dir=/home/sdl/3d_molecule_save/pretrain-$date_time/
n_gpu=1
MASTER_PORT=10086
lr=1e-4
wd=1e-4
batch_size=256
update_freq=1
heads=64
masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.30
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=1000000
contrastive_loss=30.0

# copy this file to save_dir
mkdir -p $save_dir
cp $0 $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
# torchrun --nproc_per_node=$n_gpu $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
cd /home/sdl/SDL-LNP/model/
~/miniconda3/envs/unimol/bin/python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT ~/miniconda3/envs/unimol/bin/unicore-train $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
    --num-workers 8 --ddp-backend=c10d \
    --task unimol_contrastive --loss unimol_contrastive --arch unimol_contrastive_base \
    --encoder-attention-heads $heads \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
    --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
    --update-freq $update_freq --seed $seed \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
    --max-update $max_steps --log-interval 100 --log-format simple \
    --save-interval-updates 1000 --validate-interval-updates 1000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
    --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
    --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
    --save-dir $save_dir --only-polar $only_polar --mode train --contrastive-loss $contrastive_loss 2>&1 | tee $save_dir/train.log
