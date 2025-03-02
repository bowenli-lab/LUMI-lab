#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --array=0-9
#SBATCH --mem=198G
#SBATCH --qos=cpu_qos
#SBATCH -p cpu
#SBATCH --mail-user=pangkuantony@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
module unload python

conda activate unimol
source /h/pangkuan/miniconda3/envs/unimol/bin/activate

# SLURM_ARRAY_TASK_ID=0
echo "processing ${SLURM_ARRAY_TASK_ID}"

INPUT_DIR="/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/missing_chembl_partition"
OUTPUT_DIR="/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/conformation/missing_conformer"

input_file=${INPUT_DIR}/${SLURM_ARRAY_TASK_ID}.txt
output_file=${OUTPUT_DIR}/${SLURM_ARRAY_TASK_ID}

mkdir -p ${OUTPUT_DIR}/${SLURM_ARRAY_TASK_ID}


cd /h/pangkuan/dev/SDL-LNP/model/molecule_library/
/h/pangkuan/miniconda3/envs/unimol/bin/python vector_rdkit_conformer_gen_customized_conformer.py --inpath ${input_file} --outpath ${output_file} --data_type "missing"