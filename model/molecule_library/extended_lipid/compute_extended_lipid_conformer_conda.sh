#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --job-name=conform-gen-15M
#SBATCH --array=64-127
#SBATCH --mem=198G
#SBATCH --qos=cpu_qos
#SBATCH -p cpu
#SBATCH --mail-user=subercui@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

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

# SLURM_ARRAY_TASK_ID=0
echo "processing ${SLURM_ARRAY_TASK_ID}"

INPUT_DIR="/scratch/ssd004/datasets/cellxgene/3d_molecule_data/15m-lib/partitioned_txt"
OUTPUT_DIR="/scratch/ssd004/datasets/cellxgene/3d_molecule_data/15m-lib/lmdb"

input_file=${INPUT_DIR}/${SLURM_ARRAY_TASK_ID}.txt
output_file=${OUTPUT_DIR}/${SLURM_ARRAY_TASK_ID}

mkdir -p ${OUTPUT_DIR}/${SLURM_ARRAY_TASK_ID}

cd /fs01/home/haotian/SDL-LNP/model/molecule_library/
python vector_rdkit_conformer_gen_customized_conformer.py --inpath ${input_file} --outpath ${output_file} --data_type "extended_lipid"
