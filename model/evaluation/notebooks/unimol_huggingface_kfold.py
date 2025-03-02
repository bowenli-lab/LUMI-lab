import os
import sys
from datetime import datetime
from pathlib import Path
import argparse

import numpy as np
import torch
import transformers
import wandb
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import KFold
from transformers import Trainer, TrainingArguments
from unicore.optim.fused_adam import FusedAdam

sys.path.insert(0, "../../")
from unimol.hf_unimol import UniMol, UniMolConfig, init_unimol_backbone
from unimol.hg_mterics import compute_metrics, NestedKFold, compile_test_metrics, ProgressionKFold
from unimol.lmdb_dataset import collate_fn, load_dataset, compute_hash2smi
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--weight-name", type=str, default="sdl-proj-head-ckpt-2")
parser.add_argument("--kfold-mode", type=str, default="sequential",
                    choices=["random", "sequential", "nested", "progression"])
parser.add_argument("--num-folds", type=int, default=5)
parser.add_argument("--loss-type", type=str, 
                    default="normalized_preference_log",
                    choices=["mse", # raw setting
                             "weighted_mse",  # weight on higher valued data
                             "plus_one_weighted_mse",
                             "preference", # pair-wise loss
                             "preference_log", # pair-wise loss with log-sig
                             "normalized_preference",  # pair-wise loss with normalization
                             "normalized_preference_log", # pair-wise loss with normalization and log-sig
                             "sampling_normalized_preference_log" # pair-wise loss with normalization and log-sig with sampling
                             ]
                    )
parser.add_argument("--clamp-zero", action="store_true")
parser.add_argument("--local", action="store_true")
parser.add_argument("--drop-extrema", action="store_true")
parser.add_argument("--topk-conformer", type=int, default=11)
parser.add_argument("--topk-conformer-test", type=int, default=11)
parser.add_argument("--save-model", action="store_true")
parser.add_argument("--parallel-fold", type=int, default=-1)
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--freeze-backbone", action="store_true")
parser.add_argument(
    "--lmdb-data",
    type=str,
    default=None,
    help="path to the lmdb data for fine-tuning. If provided, this will be an integrated dataset for training, validation.",
)

args, unknown = parser.parse_known_args()

# ===== config =====

if not args.local:
    with open('model_path.yaml', 'r') as file:
        weight_mapping = yaml.safe_load(file)
else:
    with open('model_path_local.yaml', 'r') as file:
        weight_mapping = yaml.safe_load(file)

if args.weight_name in weight_mapping:
    weight_path = weight_mapping[args.weight_name]
    weight_path = weight_mapping[args.weight_name]
else:
    weight_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_save/" + args.weight_name + "/checkpoint_best.pt"

weight_name = args.weight_name
# if args.loss_type == "mse":
#     weight_name = args.weight_name 
# else: 
#     weight_name = f"{args.weight_name}_{args.loss_type}"

dict_path = "./dict.txt"
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
model_config = UniMolConfig(
    input_dim=512,
    inner_dim=512 if not args.freeze_backbone else 2048,
    num_classes=1,
    dropout=0,
    decoder_type="mlp",
    loss_type=args.loss_type,
    clamp_zero=args.clamp_zero,
    dict_path=dict_path,
    backbone_path=weight_path,
    freeze_backbone=args.freeze_backbone,
)
# epoch_num = 24
epoch_num = 32
# epoch_num=2
warmup_ratio = 0.06
base_name = f"{weight_name}_{str(Path(args.lmdb_data).stem)}"
if args.ckpt_path is None:
    output_path = f"/datasets/cellxgene/3d_molecule_save/fine-tuning/{base_name}"
else:
    output_path = args.ckpt_path + f"/{base_name}"

# mkdir
os.makedirs(output_path, exist_ok=True)

# kfold_order = "sequential"  # random | sequential | nested | progression
kfold_order = args.kfold_mode

print(f"Running weight_name: {weight_name}; kfold_mode: {kfold_order}; loss_type: {args.loss_type}")


# load model and dictionary
model_backbone, dictionary = init_unimol_backbone(weight_path, dict_path=dict_path)

# build huggingface dataset from lmdb
if args.lmdb_data is not None:
    raw_combined_data = load_dataset(
        dictionary,
        args.lmdb_data,
        None,
        topk_conformer=args.topk_conformer,
    )
    hf_combined_data = Dataset.from_generator(lambda: raw_combined_data)
    combined_hash2smi = compute_hash2smi(raw_combined_data)
    num_folds = args.num_folds
    training_steps = len(hf_combined_data) * (1 - 1 / num_folds) * epoch_num
    warmup_steps = int(training_steps * warmup_ratio)

    if kfold_order == "sequential":
        kf = KFold(n_splits=num_folds, shuffle=False)
        combined_data = hf_combined_data
    else:
        raise ValueError("Only sequential kfold is supported for integrated dataset so far")

else:
    lmdb_dir = Path("/datasets/cellxgene/3d_molecule_data/1920-lib" if not args.local else "/home/sdl/vector_cellxgene_data/1920-lib")
    train_data = load_dataset(
        dictionary,
        str(lmdb_dir / "train.lmdb"),
        "train",
        topk_conformer=args.topk_conformer,
    )
    valid_data = load_dataset(
        dictionary,
        str(lmdb_dir / "valid.lmdb"),
        "valid",
        topk_conformer=args.topk_conformer,
    )
    test_data = load_dataset(
        dictionary,
        str(lmdb_dir / "test.lmdb"),
        "test",
        topk_conformer=args.topk_conformer,
    )


    hf_train_data = Dataset.from_generator(
        lambda: train_data,
    )

    hf_valid_data = Dataset.from_generator(
        lambda: valid_data,
    )

    hf_test_data = Dataset.from_generator(
        lambda: test_data,
    )

    combined_hash2smi = {**compute_hash2smi(train_data),
                        **compute_hash2smi(valid_data), 
                        **compute_hash2smi(test_data)}

    training_steps = len(hf_train_data) * epoch_num
    warmup_steps = int(training_steps * warmup_ratio)
    num_folds = args.num_folds

    if kfold_order == "random":
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        combined_data = concatenate_datasets([hf_train_data, hf_valid_data])
    elif kfold_order == "sequential":
        kf = KFold(n_splits=num_folds, shuffle=False)
        combined_data = concatenate_datasets([hf_train_data, hf_valid_data])
    elif kfold_order == "nested":
        kf = NestedKFold(n_splits=num_folds, shuffle=False)
        combined_data = concatenate_datasets([hf_train_data, hf_valid_data, hf_test_data])
    elif kfold_order == "progression":
        kf = ProgressionKFold(n_splits=num_folds, shuffle=False)
        combined_data = concatenate_datasets([hf_train_data, hf_valid_data])
    else:
        print("Invalid kfold order, using sequential instead")
        kf = KFold(n_splits=num_folds, shuffle=False)
        combined_data = concatenate_datasets([hf_train_data, hf_valid_data])


num_data = len(combined_data)
print(f"Number of data for k-fold training: {num_data}")
save_steps = 50 * max(num_data // 1750, 1)
eval_steps = save_steps // 2
# Perform cross-validation
fold_metrics = []
test_metrics = []
for fold, index_tuple in enumerate(kf.split(combined_data), 1):
    if args.parallel_fold != -1 and fold != args.parallel_fold:
        print(f"Skipping fold {fold}... due to parallel_fold option")
        continue
    
    print(f"Fold {fold}")
    if not args.freeze_backbone:
        run_name = f"{base_name}_fold_{fold}"
    else:
        run_name = f"{base_name}_freeze_backbone_fold_{fold}"
    if kfold_order == "sequential":
        project_name = "kfold"  # TODO: update project name if args.lmdb_data
        train_index, val_index = index_tuple
    elif kfold_order == "random":
        project_name = "kfold-random"
        train_index, val_index = index_tuple
    elif kfold_order == "nested":
        # project_name = "kfold-nested"
        project_name = "kfold-nested-benchmark-sweep-re"
        train_index, val_index, test_idx = index_tuple
    elif kfold_order == "progression":
        project_name = "kfold-progression-benchmark"
        train_index, val_index, test_idx = index_tuple
    else:
        raise ValueError(f"Invalid kfold order: {kfold_order}")
    
    # project_name += f"-finetune-strategy"

    if args.lmdb_data is not None:
        project_name = str(Path(args.lmdb_data).stem) + "-" + project_name
    else:
        project_name = "1920-model-" + project_name
    
    if args.save_model:
        project_name += "-save-model"
        
    
    fold_output_path = f"{output_path}/{run_name}"
    print(f"Fold {fold} output path: {fold_output_path}")
    os.makedirs(fold_output_path, exist_ok=True)
    
    training_args = TrainingArguments(
            output_dir=fold_output_path,
            num_train_epochs=epoch_num,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=256,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            logging_dir="./logs",
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=save_steps,
            eval_steps=eval_steps,
            report_to="wandb",
            label_names=["target", "smi_name", ],
            load_best_model_at_end=True,
            optim="adamw_torch",
            metric_for_best_model="relaxed_spearman",
            learning_rate=0.00005 if not args.freeze_backbone else 5e-4,
        )

        
    with wandb.init(project=project_name, name=run_name, config=training_args):
        
    
        
        # wandb config
        wandb.config.update(
            {
                "weight_path": weight_path,
                "kfold_order": kfold_order,
                "weight_name": weight_name,
                "fold": fold,
                "loss_type": args.loss_type,
                "clamp_zero": model_config.clamp_zero,
                "drop_extrema": args.drop_extrema,
            }
        )
        
        train_data = combined_data.select(train_index)
        val_data = combined_data.select(val_index)
        if kfold_order == "nested" or kfold_order == "progression":
            test_data = combined_data.select(test_idx)
        else:
            test_data = val_data if args.lmdb_data else hf_test_data
        
        # load model and dictionary
        model_backbone, dictionary = init_unimol_backbone(
            weight_path=weight_path, dict_path=dict_path
        )

        # %%
        # Reinitialize model config and model for each fold
        model_config = UniMolConfig(
            input_dim=512,
            inner_dim=512 if not args.freeze_backbone else 2048,
            num_classes=1,
            dropout=0,
            decoder_type="mlp",
            loss_type=args.loss_type,
            clamp_zero=args.clamp_zero,
            dict_path=dict_path,
            backbone_path=weight_path,
            freeze_backbone=args.freeze_backbone,
        )

        model = UniMol(model_config)
        model.init_backbone(model_backbone)
        # Initialize a new model for each fold

        optimizer = FusedAdam(
            model.parameters(),
            lr=3e-5,
            eps=1e-6,
            betas=(0.9, 0.99),
        )

        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )

        # Create a new trainer for each fold
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=collate_fn,
            compute_metrics=compute_metrics(smi_string = None, 
                                            drop_extrema=args.drop_extrema,
                                            keep_topk_conformer=args.topk_conformer),
            tokenizer=None,
            optimizers=(optimizer, scheduler),
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

        # Train the model for the current fold
        trainer.train()

        # Evaluate the model on the validation set for the current fold
        output = trainer.predict(val_data, metric_key_prefix="eval").metrics
        print(f"Fold {fold} validation metrics: {output}")
        fold_metrics.append(output)

        output = trainer.predict(test_data, metric_key_prefix="test")
        test_output = output.metrics
        
        compile_test_metrics(output, combined_hash2smi, select_topk_conformer=args.topk_conformer_test)
            
        print(f"Fold {fold} test metrics: {test_output}")
        test_metrics.append(test_output)

        wandb.log(test_output)
        
        # upload the model to hub
        if args.save_model:
            print(f"Saving model to {fold_output_path}")
            model.save_pretrained(fold_output_path)
            model.push_to_hub(run_name, token=os.environ.get("HUGGINGFACE_TOKEN", None))
        
        print(f"Fold {fold} done, cleaning up...")
        os.system(f"rm -rf {fold_output_path}")

# Calculate average metrics across all folds
avg_metrics = {
    metric: sum(fold[metric] for fold in fold_metrics) / num_folds
    for metric in fold_metrics[0]
}

test_avg_metrics = {
    metric: sum(fold[metric] for fold in test_metrics) / num_folds
    for metric in test_metrics[0]
}
print(f"Average validation metrics across all folds: {avg_metrics}")

print(f"Average test metrics across all folds: {test_avg_metrics}")



wandb.finish()
