# %%
# %load_ext autoreload
# %autoreload 2
from typing import Dict
from datasets import Dataset
import sys
import os
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    TrainingArguments,
    PretrainedConfig,
)
import wandb
from fused_adam import FusedAdam
import transformers
from transformers.activations import ACT2CLS
from transformers import EarlyStoppingCallback

sys.path.insert(
    0,
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/",
)

from torch.nn.utils.rnn import pad_sequence
from torch import tensor
from unimol.models import UniMolModel, unimol_base_architecture
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
)
from unimol.data.tta_dataset import TTADataset

import argparse
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn
import pytorch_warmup as warmup

import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem import Draw

# RDLogger.DisableLog('rdApp.*')
import warnings

warnings.filterwarnings(action="ignore")
from multiprocessing import Pool


# %%

import argparse
from unicore.data import Dictionary

from unicore.data import UnicoreDataset
import numpy as np
from scipy.stats import rankdata
from torch.utils.data.dataloader import default_collate


# Define the hyperparameter sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval/relaxed_spearman", "goal": "maximize"},
}


# hyperparameters
parameters_dict = {
    "epochs": {"distribution": "int_uniform", "min": 10, "max": 80},
    "batch_size": {"values": [8, 16, 32, 64, 128, 256]},
    "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
    "weight_decay": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
    "warmup_ratio": {"distribution": "uniform", "min": 0.0, "max": 0.1},
    "seed": {
        "values": [
            0,
        ]
    },
    "dropout": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
    "inner_dim": {"values": [256, 512, 1024]},
    "lr_scheduler": {
        "values": ["linear", "cosine", "cosine_with_restarts", "polynomial"]
    },
    "adam_epsilon": {"values": [1e-8, 1e-6, 1e-4]},
    "adam_beta1": {"values": [0.9, 0.95, 0.99]},
    "adam_beta2": {"values": [0.999, 0.9999, 0.99999]},
    "activation_fn": {"values": ["gelu", "mish", "linear", "tanh"]},
}


sweep_config["parameters"] = parameters_dict
sweep_config["early_terminate"] = {"type": "hyperband", "min_iter": 3}


def relaxed_spearman_correlation(preds, targets, relax_ratio=0.05):
    """
    Compute the relaxed spearman correlation. The relax ratio tells the amount of difference allowed. A delta threshold will be computed by the relax ratio times the dynamica range of the target values. For each pair of values, if the difference is smaller than the delta threshold, will make the difference to be zero.

    Args:
        preds (np.ndarray): The predicted values.
        target (np.ndarray): The target values.
        relax_ratio (float): The relax ratio.
    """

    assert len(preds) == len(
        targets
    ), "The length of preds and target should be the same."
    n = len(preds)

    # Use rankdata to correctly handle ties
    x_rank = rankdata(preds)
    y_rank = rankdata(targets)
    delta = relax_ratio * n

    # Calculate the difference in ranks
    d = x_rank - y_rank
    d = np.where(np.abs(d) <= delta, 0, d)

    # Calculate the sum of the squared differences
    d_squared_sum = np.sum(d**2)

    # Calculate the Spearman correlation coefficient
    correlation = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return correlation


class HashArrayDataset(UnicoreDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        output = hash(self.dataset[index])
        return output

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return default_collate(samples)


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        # TDOO:
        # idx = 1
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data


# %%

NUM_CONFORMER = 11


def load_dataset(dictionary, split_path, split):
    """Load a given dataset split.
    Args:
        split (str): name of the data scoure (e.g., train)
    """

    dataset = LMDBDataset(split_path)

    dataset = TTADataset(dataset, 0, "atoms", "coordinates", NUM_CONFORMER)
    dataset = AtomTypeDataset(dataset, dataset)
    tgt_dataset = KeyDataset(dataset, "target")
    smi_dataset = KeyDataset(dataset, "smi")

    dataset = RemoveHydrogenDataset(
        dataset,
        "atoms",
        "coordinates",
        True,
        True,
    )
    dataset = CroppingDataset(dataset, 0, "atoms", "coordinates", 512)
    dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
    src_dataset = KeyDataset(dataset, "atoms")
    src_dataset = TokenizeDataset(src_dataset, dictionary, max_seq_len=512)
    coord_dataset = KeyDataset(dataset, "coordinates")

    def PrependAndAppend(dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)

    src_dataset = PrependAndAppend(src_dataset, dictionary.bos(), dictionary.eos())
    edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
    coord_dataset = FromNumpyDataset(coord_dataset)
    coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
    distance_dataset = DistanceDataset(coord_dataset)

    nest_dataset = NestedDictionaryDataset(
        {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=dictionary.pad(),
            ),
            "src_coord": RightPadDatasetCoord(
                coord_dataset,
                pad_idx=0,
            ),
            "src_distance": RightPadDataset2D(
                distance_dataset,
                pad_idx=0,
            ),
            "src_edge_type": RightPadDataset2D(
                edge_type,
                pad_idx=0,
            ),
            "target": RawLabelDataset(tgt_dataset),
            "smi_name": HashArrayDataset(smi_dataset),
        },
    )
    datasets = nest_dataset
    return datasets


def collate_fn(batch):
    # Initialize dictionaries to hold processed batch data
    net_input = {
        "src_tokens": [],
        "src_coord": [],
        "src_distance": [],
        "src_edge_type": [],
    }
    target = {"target": []}
    smi_name = []

    # Loop over all data points in the batch
    for item in batch:
        for key in net_input:
            net_input[key].append(tensor(item[f"{key}"]))
        target["target"].append(tensor(item["target"]))
        smi_name.append(torch.tensor(item["smi_name"]))

    # Pad sequences where necessary
    # pad src_tokens
    key = "src_tokens"

    net_input[key] = pad_sequence(net_input[key], batch_first=True, padding_value=0)

    key = "src_coord"
    max_length = max(x.size(0) for x in net_input[key])
    max_width = max(x.size(1) for x in net_input[key])
    net_input[key] = [
        torch.nn.functional.pad(
            x, (0, max_width - x.size(1), 0, max_length - x.size(0)), "constant", 0
        )
        for x in net_input[key]
    ]
    net_input[key] = torch.stack(net_input[key])

    key = "src_edge_type"
    max_length = max(x.size(0) for x in net_input[key])
    max_width = max(x.size(1) for x in net_input[key])
    net_input[key] = [
        torch.nn.functional.pad(
            x, (0, max_width - x.size(1), 0, max_length - x.size(0)), "constant", 0
        )
        for x in net_input[key]
    ]
    net_input[key] = torch.stack(net_input[key])

    key = "src_distance"
    max_length = max(x.size(0) for x in net_input[key])
    max_width = max(x.size(1) for x in net_input[key])
    net_input[key] = [
        torch.nn.functional.pad(
            x, (0, max_width - x.size(1), 0, max_length - x.size(0)), "constant", 0
        )
        for x in net_input[key]
    ]
    net_input[key] = torch.stack(net_input[key])

    # Convert lists to tensors
    target = torch.stack(target["target"])
    smi_name = torch.stack(smi_name)

    # Return the collated batch
    return {"target": target, "smi_name": smi_name, **net_input}


dictionary = Dictionary().load(
    "/home/pangkuan/dev/SDL-LNP/model/unimol/notebooks/dict.txt"
)
weight_path = "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/model_weight/mol_pre_no_h_220816.pt"
dictionary.add_symbol("[MASK]", is_special=True)

# build huggingface dataset from lmdb
train_data = load_dataset(
    dictionary,
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/4CR_customized/train.lmdb",
    "train",
)
valid_data = load_dataset(
    dictionary,
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/4CR_customized/valid.lmdb",
    "valid",
)
test_data = load_dataset(
    dictionary,
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/4CR_customized/test.lmdb",
    "test",
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


# %%
class UniMolConfig(PretrainedConfig):
    def __init__(
        self,
        input_dim=512,
        inner_dim=512,
        num_classes=1,
        dropout=0.1,
        act_f="gelu",
        **kwargs,
    ):
        super(UniMolConfig, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation_fn = act_f


class UniMol(PreTrainedModel):
    config_class = UniMolConfig

    def __init__(self, model, config):
        super(UniMol, self).__init__(config=config)
        self.model = model
        self.dictionary = dictionary
        self.dense = nn.Linear(config.input_dim, config.inner_dim)
        self.activation_fn = ACT2CLS[config.activation_fn]()
        self.out_proj = nn.Linear(config.inner_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(
        self,
    ):
        # init the  dense, out_proj
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        src_tokens,
        src_coord,
        src_edge_type,
        src_distance,
        target=None,
        smi_name=None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        output = self.model(
            src_tokens=src_tokens.to(device),
            src_edge_type=src_edge_type.to(device),
            src_distance=src_distance.to(device),
            src_coord=src_coord.to(device),
        )
        (encoder_rep, encoder_pair_rep) = output
        x = encoder_rep[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        loss = None
        if target is not None:
            loss = F.mse_loss(x.squeeze(-1), target.squeeze(-1))

        return {
            "loss": loss,
            "logits": x,
            "encoder_rep": encoder_rep,
        }


def compute_metrics(pred):
    target = pred.label_ids[0].squeeze()
    prediction = pred.predictions[0].squeeze()
    smi_name = pred.label_ids[1].squeeze()

    df = pd.DataFrame(
        {
            "smi_name": smi_name,
            "target": target,
            "prediction": prediction,
        }
    )

    df = df.groupby("smi_name").mean().reset_index()

    return {
        "pearson": pearsonr(df["target"].values, df["prediction"].values)[0],
        "spearman": spearmanr(df["target"].values, df["prediction"].values)[0],
        "relaxed_spearman": relaxed_spearman_correlation(
            df["target"].values, df["prediction"].values
        ),
    }


sweep_id = wandb.sweep(sweep_config, project="unimol-sweep-relaxed-spearman")


def main(config=None):
    with wandb.init(config=config):
        config = wandb.config
        parser = argparse.ArgumentParser()

        args = parser.parse_args()

        unimol_base_architecture(args)
        model_backbone = UniMolModel(args, dictionary=dictionary)
        model_weight = torch.load(weight_path)["model"]
        model_backbone.load_state_dict(model_weight, strict=False)

        # %%
        # fixed seed
        seed = config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # %%
        from datetime import datetime

        run_name = f"4CR_customized_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # %%

        model_config = UniMolConfig(
            input_dim=512,
            inner_dim=config.inner_dim,
            num_classes=1,
            dropout=config.dropout,
            act_f=config.activation_fn,
        )

        model = UniMol(model_backbone, model_config)

        optimizer = FusedAdam(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
            betas=(config.adam_beta1, config.adam_beta2),
        )

        epoch_num = config.epochs
        warmup_ratio = config.warmup_ratio
        training_steps = len(hf_train_data) * epoch_num
        warmup_steps = int(training_steps * warmup_ratio)

        scheduler_dict = {
            "linear": transformers.get_linear_schedule_with_warmup,
            "cosine": transformers.get_cosine_schedule_with_warmup,
            "cosine_with_restarts": transformers.get_cosine_with_hard_restarts_schedule_with_warmup,
            "polynomial": transformers.get_polynomial_decay_schedule_with_warmup,
        }

        scheduler = scheduler_dict[config.lr_scheduler](
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )

        training_args = TrainingArguments(
            output_dir="/home/pangkuan/dev/dd_1/uni-mol-model/results",
            num_train_epochs=epoch_num,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=64,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            logging_dir="./logs",
            fp16=True,
            logging_steps=25,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=400,
            eval_steps=400,
            report_to="wandb",
            run_name=run_name,
            label_names=["target", "smi_name"],
            load_best_model_at_end=True,
            optim="adamw_torch",
            metric_for_best_model="relaxed_spearman",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=hf_train_data,
            eval_dataset=hf_valid_data,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler),
            tokenizer=None,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

        trainer.train()

        output = trainer.predict(
            hf_test_data,
            metric_key_prefix="test",
        )
        print(output.metrics)
        wandb.log(output.metrics)

        trainer.save_model(f"/home/pangkuan/dev/dd_1/uni-mol-model/{run_name}")


wandb.agent(sweep_id, function=main, count=150)
