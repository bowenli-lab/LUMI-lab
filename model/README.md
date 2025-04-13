# Model Directory

This directory contains the core model implementations and utilities for LUMI-lab. It integrates various modules for model training, evaluation, and active learning.

## Directory Structure

- **active_learning/**  
  Contains scripts and notebooks for active learning experiments and strategies.

- **data_process/**  
  Utilities and scripts for processing and preparing datasets.

- **evaluation/**  
  Notebooks and scripts used for model evaluation, fine-tuning, and inference.

- **molecule_library/**  
  Tools for molecular data management, including conformer generation.

- **pretrain/**  
  Code related to the pretraining of foundation models.

- **serverless/**  
  Implements serverless deployments and endpoints for parallel inference with [Modal](https://modal.com/).

- **unimol/**  
  Contains the UniMol-based model implementations, we include modified archetectures in the it.
  
## Usage
- For molecule library generation, refer to:
  - `/model/molecule_library/submit_rdkit_conformer.sh`
- For pretraining and continual pretraining:
  - `/model/pretrain/pretrain.sh`
  - `/model/pretrain/continual_pretrain.sh`
- For finetuning and active learning:
  - finetuning example: `/model/evaluation/notebooks/finetune.sh`
  - active learning example: `/active_learning/propose_all_tops.ipynb`
