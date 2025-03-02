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

# Change to the directory containing the notebook
cd /home/sdl/SDL-LNP/model/evaluation/notebooks

model_name="sdl-contrastive-continual-ckpt_2024-07-08_12-10-42_fold_1"
# model_name="baseline_2024-06-10_11-43-48_fold_5"

# Run the inference script
~/miniconda3/envs/unimol/bin/python /home/sdl/SDL-LNP/model/evaluation/notebooks/unimol_inference.py \
    --backbone-path "/home/sdl/3d_molecule_save/baseline/10x-sdl-contrastive-ckpt-3/checkpoint_last.pt" \
    --dict-path "./dict.txt" \
    --pretrained-weight "KuanP/${model_name}" \
    --lmdb-dir "/home/sdl/vector_cellxgene_data/220k-lib/all-lmdb" \
    --result-csv "test_result.csv" \
    --eval-batch-size 256 \
    --smi-path "smi_name.txt" \
    --shard-id -1 \
    --num-shards -1 \
    --return-representation \
    --load-from-local
    
rm -rf "KuanP/"
