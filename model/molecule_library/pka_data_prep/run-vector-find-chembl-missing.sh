#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=64
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


cd /h/pangkuan/dev/SDL-LNP/model/molecule_library/pka_data_prep/
/h/pangkuan/miniconda3/envs/unimol/bin/python find_chembl_missing_conformer.py