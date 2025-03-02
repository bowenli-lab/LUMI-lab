#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --mail-user=pangkuantony@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --account=def-bowenli

module load StdEnv/2020


export SLURM_TMPDIR=$SLURM_TMPDIR
cd /home/pangkuan/projects/def-bowenli/pangkuan/SDL-LNP/model/molecule_library/extended_lipid
~/miniconda3/envs/unimol/bin/python merge_lmdb.py