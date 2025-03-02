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

# export HF_DATASETS_CACHE="/scratch/ssd004/scratch/pangkuan/hf_cache"

weight_name="sdl-contrastive-continual-ckpt"

cd /home/sdl/SDL-LNP/model/evaluation/notebooks
~/miniconda3/envs/unimol/bin/python /home/sdl/SDL-LNP/model/evaluation/notebooks/unimol_huggingface_kfold.py --weight-name ${weight_name} --local