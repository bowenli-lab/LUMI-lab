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
from transformers import PretrainedConfig, PreTrainedModel
from torch.nn import functional as F

sys.path.insert(0, "../../")
from unimol.hf_unimol import UniMol, UniMolConfig, init_unimol_backbone
from unimol.hg_mterics import compute_metrics, NestedKFold, compile_test_metrics, ProgressionKFold
from unimol.lmdb_dataset import collate_fn_molecular_descriptor, load_dataset_molecular_descriptor, compute_hash2smi, load_dataset
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--weight-name", type=str, default="molecular-descriptor")
parser.add_argument("--kfold-mode", type=str, default="nested",
                    choices=["random", "sequential", "nested", "progression"])
parser.add_argument("--loss-type", type=str, default="mse",
                    choices=["mse", "flooding", "weighted_mse"])
parser.add_argument("--clamp-zero", action="store_true")
args, unknown = parser.parse_known_args()



# ===== MLP model =====
# in hf
class MLPConfig(PretrainedConfig):
    def __init__(self, input_dim=1613, inner_dim=512,
                 num_classes=1, layers=3, dropout=0.1, **kwargs):
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.layers = layers
        self.clamp_zero = kwargs.pop("clamp_zero", False)

class MLPDecoder(PreTrainedModel):
    config_class = MLPConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(config.input_dim, config.inner_dim))
        for _ in range(config.layers - 1):
            self.layers.append(torch.nn.Linear(config.inner_dim, config.inner_dim))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(config.inner_dim, config.num_classes))
        self.dropout = torch.nn.Dropout(config.dropout)
        self.apply(self._init_weights)
        print(self)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, molecular_descriptor, target=None, smi_name=None):
        x = molecular_descriptor        
        if torch.isnan(x).any():
            raise ValueError("x still has nan after filling")
        

        for i, layer in enumerate(self.layers):
            x = self.dropout(x)
            x = layer(x)
            if torch.isnan(x).any():
                print(f"Layer {i} weights: {layer.weight.data}")
                print(f"Layer {i} bias: {layer.bias.data}")
                raise ValueError(f"NaN detected after layer {i}")
        
        logits = x
        loss = None
        if target is not None:
            if self.config.clamp_zero:
                target = torch.clamp(target, min=0)
            loss = F.mse_loss(x.squeeze(-1), target.squeeze(-1))
        
        # print(f"logits: {logits}")
        return {
            "loss": loss,
            "logits": logits,
            "encoder_rep": x,
        }
    
# ===== config =====

with open('model_path.yaml', 'r') as file:
    weight_mapping = yaml.safe_load(file)

weight_path = weight_mapping[args.weight_name]

weight_name = args.weight_name

dict_path = "./dict.txt"
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
model_config = MLPConfig(
    inner_dim=512,
    layers=3)
# epoch_num = 24
epoch_num = 32
# epoch_num=2
warmup_ratio = 0.01
output_path = "/datasets/cellxgene/3d_molecule_save/fine-tuning"
# kfold_order = "sequential"  # random | sequential | nested | progression
kfold_order = args.kfold_mode

print(f"Running weight_name: {weight_name}; kfold_mode: {kfold_order}; loss_type: {args.loss_type}")



# build huggingface dataset from lmdb
lmdb_dir = Path("/datasets/cellxgene/3d_molecule_data/1920-lib")
train_data = load_dataset_molecular_descriptor(
    str(lmdb_dir / "train.lmdb"),
    "train",
)
valid_data = load_dataset_molecular_descriptor(
    str(lmdb_dir / "valid.lmdb"),
    "valid",
)
test_data = load_dataset_molecular_descriptor(
    str(lmdb_dir / "test.lmdb"),
    "test",
)

model_backbone, dictionary = init_unimol_backbone(weight_path, dict_path=dict_path)
train_data_raw = load_dataset(
    dictionary,
    str(lmdb_dir / "train.lmdb"),
    "train",
)

valid_data_raw = load_dataset(
    dictionary,
    str(lmdb_dir / "valid.lmdb"),
    "valid",
)

test_data_raw = load_dataset(
    dictionary,
    str(lmdb_dir / "test.lmdb"),
    "test",
)

def shard_loader(data_lst, idx_lsts):
    data = data_lst[0]
    for idx_lst in idx_lsts:
        for idx in idx_lst:
            pass
            yield data[idx] 
N = os.cpu_count()

train_shard_idx_lst = list(range(len(train_data))) 
valid_shard_idx_lst = list(range(len(valid_data)))
test_shard_idx_lst = list(range(len(test_data)))
# partition to multiple continuous shards
train_shard_idx_lst = [train_shard_idx_lst[i:i+int(len(train_shard_idx_lst)/N)] for i in range(0, len(train_shard_idx_lst), int(len(train_shard_idx_lst)/N) )]
valid_shard_idx_lst = [valid_shard_idx_lst[i:i+int(len(valid_shard_idx_lst)/N)] for i in range(0, len(valid_shard_idx_lst), int(len(valid_shard_idx_lst)/N))]
test_shard_idx_lst = [test_shard_idx_lst[i:i+int(len(test_shard_idx_lst)/N)] for i in range(0, len(test_shard_idx_lst), int(len(test_shard_idx_lst)/N))]


hf_train_data = Dataset.from_generator(
    shard_loader,
    gen_kwargs={
        "data_lst": [train_data for _ in range(N)],
        "idx_lsts": train_shard_idx_lst,
    },
    num_proc=6,
)

print("train data loaded")

hf_valid_data = Dataset.from_generator(
    shard_loader,
    gen_kwargs={
        "data_lst": [valid_data for _ in range(N)],
        "idx_lsts": valid_shard_idx_lst,
    },
    num_proc=6,

)
print("valid data loaded")

hf_test_data = Dataset.from_generator(
    shard_loader,
    gen_kwargs={
        "data_lst": [test_data for _ in range(N)],
        "idx_lsts": test_shard_idx_lst,
    },
    num_proc=6,
)
print("test data loaded")

combined_hash2smi = {**compute_hash2smi(train_data_raw),
                     **compute_hash2smi(valid_data_raw), 
                     **compute_hash2smi(test_data_raw)}
print(f"Combined data size: {len(hf_train_data) + len(hf_valid_data) + len(hf_test_data)}")

training_steps = len(hf_train_data) * epoch_num
warmup_steps = int(training_steps * warmup_ratio)
print(f"Training steps: {training_steps}, Warmup steps: {warmup_steps}")

training_args = TrainingArguments(
    output_dir=output_path,
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
    save_steps=500,
    eval_steps=250,
    report_to="wandb",
    label_names=["target", "smi_name", ],
    load_best_model_at_end=True,
    optim="adamw_torch",
    metric_for_best_model="relaxed_spearman",
)


num_folds = 5

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
    kf = KFold(n_splits=num_folds, shuffle=False)
    combined_data = concatenate_datasets([hf_train_data, hf_valid_data])




# Perform cross-validation
fold_metrics = []
test_metrics = []
base_name = f"{weight_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
for fold, index_tuple in enumerate(kf.split(combined_data), 1):
    print(f"Fold {fold}")
    run_name = f"{base_name}_fold_{fold}"
    run_name = f"{base_name}_fold_{fold}"
    if kfold_order == "sequential":
        project_name = "1920-model-kfold"
        train_index, val_index = index_tuple
    elif kfold_order == "random":
        project_name = "1920-model-kfold-random"
        train_index, val_index = index_tuple
    elif kfold_order == "nested":
        project_name = "1920-model-kfold-nested"
        train_index, val_index, test_idx = index_tuple
    elif kfold_order == "progression":
        project_name = "1920-model-kfold-progression"
        train_index, val_index, test_idx = index_tuple
    else:
        raise ValueError(f"Invalid kfold order: {kfold_order}")
        
    with wandb.init(project=project_name, name=run_name, config=training_args):
        
        # wandb config
        wandb.config.update(
            {
                "kfold_order": kfold_order,
                "weight_name": weight_name,
                "fold": fold,
                "loss_type": args.loss_type,
                "clamp_zero": model_config.clamp_zero,
            }
        )
        
        train_data = combined_data.select(train_index)
        val_data = combined_data.select(val_index)
        if kfold_order == "nested" or kfold_order == "progression":
            test_data = combined_data.select(test_idx)
        

        # %%
        # Initialize a new model for each fold
        model = MLPDecoder(model_config)

        optimizer = FusedAdam(
            model.parameters(),
            lr=1e-4,
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
            data_collator=collate_fn_molecular_descriptor,
            compute_metrics=compute_metrics(smi_string = None),
            tokenizer=None,
            optimizers=(optimizer, scheduler),
        )

        # Train the model for the current fold
        trainer.train()

        # Evaluate the model on the validation set for the current fold
        output = trainer.predict(val_data, metric_key_prefix="eval").metrics
        print(f"Fold {fold} validation metrics: {output}")
        fold_metrics.append(output)

        output = trainer.predict(hf_test_data, metric_key_prefix="test")
        test_output = output.metrics
        
        compile_test_metrics(output, combined_hash2smi)
            
        print(f"Fold {fold} test metrics: {test_output}")
        test_metrics.append(test_output)

        wandb.log(test_output)

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
