import os
import numpy as np
import torch
import yaml
import wandb
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Type, Union, List, Optional

from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import KFold
from transformers import Trainer, TrainingArguments
import transformers

from unimol.hf_unimol import UniMol, UniMolConfig, init_unimol_backbone
from unimol.hg_metrics import compute_metrics, NestedKFold, compile_test_metrics, ProgressionKFold
from unimol.lmdb_dataset import collate_fn, load_dataset, compute_hash2smi
from unicore.optim.fused_adam import FusedAdam


class ModelManager:
    def __init__(self, config_path: str, weight_name: str, kfold_mode: str, loss_type: str, save_model: bool, local: bool = False):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.weight_path = self.config[weight_name]
        self.kfold_mode = kfold_mode
        self.loss_type = loss_type
        self.save_model = save_model
        self.local = local

        self.model_config = UniMolConfig(
            input_dim=512,
            inner_dim=512,
            num_classes=1,
            dropout=0,
            decoder_type="mlp",
            loss_type=self.loss_type,
            clamp_zero=False,
            dict_path="./dict.txt",
            backbone_path=self.weight_path,
        )
        
        self.training_args = TrainingArguments(
            output_dir=self._get_output_path(),
            num_train_epochs=32,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=256,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            logging_dir="./logs",
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=500,
            eval_steps=250,
            report_to="wandb",
            label_names=["target", "smi_name"],
            load_best_model_at_end=True,
            optim="adamw_torch",
            metric_for_best_model="relaxed_spearman",
        )

    def _get_output_path(self):
        base_name = f"{self.weight_path}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        output_path = f"/datasets/cellxgene/3d_molecule_save/fine-tuning/{base_name}"
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def _load_data(self):
        lmdb_dir = Path("/datasets/cellxgene/3d_molecule_data/1920-lib" if not self.local else "/home/sdl/vector_cellxgene_data/1920-lib")
        train_data = load_dataset(
            dictionary,
            str(lmdb_dir / "train.lmdb"),
            "train",
            topk_conformer=11,
        )
        valid_data = load_dataset(
            dictionary,
            str(lmdb_dir / "valid.lmdb"),
            "valid",
            topk_conformer=11,
        )
        test_data = load_dataset(
            dictionary,
            str(lmdb_dir / "test.lmdb"),
            "test",
            topk_conformer=11,
        )

        hf_train_data = Dataset.from_generator(lambda: train_data)
        hf_valid_data = Dataset.from_generator(lambda: valid_data)
        hf_test_data = Dataset.from_generator(lambda: test_data)

        combined_data = concatenate_datasets([hf_train_data, hf_valid_data, hf_test_data])

        return combined_data, hf_test_data

    def _init_kfold(self, combined_data):
        num_folds = 5
        if self.kfold_mode == "random":
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        elif self.kfold_mode == "sequential":
            kf = KFold(n_splits=num_folds, shuffle=False)
        elif self.kfold_mode == "nested":
            kf = NestedKFold(n_splits=num_folds, shuffle=False)
        elif self.kfold_mode == "progression":
            kf = ProgressionKFold(n_splits=num_folds, shuffle=False)
        else:
            kf = KFold(n_splits=num_folds, shuffle=False)
        
        return kf.split(combined_data)

    def fit(self):
        combined_data, hf_test_data = self._load_data()
        kf = self._init_kfold(combined_data)
        
        fold_metrics = []
        test_metrics = []

        for fold, index_tuple in enumerate(kf, 1):
            print(f"Fold {fold}")
            run_name = f"{self.weight_path}_fold_{fold}"

            with wandb.init(project="model_project", name=run_name, config=self.training_args):
                train_index, val_index = index_tuple[:2]
                train_data = combined_data.select(train_index)
                val_data = combined_data.select(val_index)

                model_backbone, dictionary = init_unimol_backbone(
                    weight_path=self.weight_path, dict_path="./dict.txt"
                )

                model = UniMol(self.model_config)
                model.init_backbone(model_backbone)

                optimizer = FusedAdam(
                    model.parameters(),
                    lr=1e-4,
                    eps=1e-6,
                    betas=(0.9, 0.99),
                )

                scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(len(train_data) * 32 * 0.06),
                    num_training_steps=len(train_data) * 32,
                )

                trainer = Trainer(
                    model=model,
                    args=self.training_args,
                    train_dataset=train_data,
                    eval_dataset=val_data,
                    data_collator=collate_fn,
                    compute_metrics=compute_metrics(smi_string=None),
                    tokenizer=None,
                    optimizers=(optimizer, scheduler),
                )

                trainer.train()
                val_metrics = trainer.evaluate(val_data)
                fold_metrics.append(val_metrics)

                test_metrics = trainer.evaluate(hf_test_data)
                test_metrics.append(test_metrics)

                if self.save_model:
                    model.save_pretrained(f"{self.training_args.output_dir}/{run_name}")

        avg_metrics = {
            metric: np.mean([fold[metric] for fold in fold_metrics])
            for metric in fold_metrics[0]
        }

        test_avg_metrics = {
            metric: np.mean([fold[metric] for fold in test_metrics])
            for metric in test_metrics[0]
        }
        print(f"Average validation metrics across all folds: {avg_metrics}")
        print(f"Average test metrics across all folds: {test_avg_metrics}")

    def predict(self, loader):
        model_backbone, dictionary = init_unimol_backbone(
            weight_path=self.weight_path, dict_path="./dict.txt"
        )
        
        model = UniMol(self.model_config)
        model.init_backbone(model_backbone)

        trainer = Trainer(
            model=model,
            args=self.training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics(smi_string=None),
            tokenizer=None,
        )
        
        predictions = trainer.predict(loader)
        return predictions

    def __call__(self, loader):
        return self.predict(loader)
