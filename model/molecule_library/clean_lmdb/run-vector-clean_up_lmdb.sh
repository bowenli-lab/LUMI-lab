#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=64
#SBATCH --job-name=sdl-clean-ligands
#SBATCH --mem=198G
#SBATCH --qos=cpu_qos
#SBATCH -p cpu
#SBATCH --mail-user=subercui@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

bash ~/.bashrc
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

cd ~/SDL-LNP/model/molecule_library/clean_lmdb/run-vector-clean_up_lmdb.sh

python clean_up_lmdb.py
