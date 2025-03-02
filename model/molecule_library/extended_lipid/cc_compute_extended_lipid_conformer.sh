#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=40
#SBATCH --array=0-255
#SBATCH --mem=86G
#SBATCH --mail-user=pangkuantony@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --account=def-bowenli
module unload python


# SLURM_ARRAY_TASK_ID=0
echo "processing ${SLURM_ARRAY_TASK_ID}"

INPUT_DIR="/home/pangkuan/projects/def-bowenli/pangkuan/data/15m-lib/partitioned_txt"
OUTPUT_DIR="/home/pangkuan/projects/def-bowenli/pangkuan/data/15m-lib/lmdb"

input_file=${INPUT_DIR}/${SLURM_ARRAY_TASK_ID}.txt
output_file=${OUTPUT_DIR}/${SLURM_ARRAY_TASK_ID}

mkdir -p ${OUTPUT_DIR}/${SLURM_ARRAY_TASK_ID}


cd /home/pangkuan/projects/def-bowenli/pangkuan/SDL-LNP/model/molecule_library/
~/miniconda3/envs/unimol/bin/python vector_rdkit_conformer_gen_customized_conformer.py --inpath ${input_file} --outpath ${output_file} --data_type "extended_lipid"