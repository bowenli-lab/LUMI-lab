# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F

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

# %%
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


# write_lmdb(inpath='/home/pangkuan/dev/SDL-LNP/model/4CR-1920.csv', outpath='/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/data/lmdb', nthreads=8)

# %%

import argparse
from unicore.data import Dictionary

parser = argparse.ArgumentParser()
dictionary = Dictionary().load(
    "/home/pangkuan/dev/SDL-LNP/model/unimol/notebooks/dict.txt"
)
dictionary.add_symbol("[MASK]", is_special=True)
# UniMolModel.add_args(parser)
args = parser.parse_args()

unimol_base_architecture(args)
model_backbone = UniMolModel(args, dictionary=dictionary)
print("Done")

# %%
# load weight
weight_path = "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/model_weight/mol_pre_no_h_220816.pt"
# weight_path = "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/model_weight/customized_model/checkpoint_best.pt"


model_weight = torch.load(weight_path)["model"]
model_backbone.load_state_dict(model_weight, strict=False)

# %%
# fixed seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
# %%


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
    if split == "train":
        tgt_dataset = KeyDataset(dataset, "target")
        smi_dataset = KeyDataset(dataset, "smi")
        sample_dataset = ConformerSampleDataset(dataset, 0, "atoms", "coordinates")
        dataset = AtomTypeDataset(dataset, sample_dataset)
    else:
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
            "net_input": {
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
            },
            "target": {
                "finetune_target": RawLabelDataset(tgt_dataset),
            },
            "smi_name": RawArrayDataset(smi_dataset),
        },
    )
    if split == "train":
        with data_utils.numpy_seed(0):
            shuffle = np.random.permutation(len(src_dataset))

        datasets = SortDataset(
            nest_dataset,
            sort_order=[shuffle],
        )
    else:
        datasets = nest_dataset
    return datasets


# %%
class UniMol(nn.Module):
    def __init__(
        self,
        model,
        input_dim=512,
        inner_dim=512,
        num_classes=1,
        activation_fn="gelu",
        dropout=0.0,
    ):
        super(UniMol, self).__init__()
        self.model = model
        self.dictionary = dictionary
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.GELU()
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        net_input = data["net_input"]
        src_tokens = net_input["src_tokens"]
        src_coord = net_input["src_coord"]
        src_edge_type = net_input["src_edge_type"]
        src_distance = net_input["src_distance"]

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
        return x


# %%
model = UniMol(
    model_backbone,
)

# %%
# get dataset

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

# get length
print(len(train_data))
print(len(valid_data))
print(len(test_data))


def collate_fn(batch):
    # Initialize dictionaries to hold processed batch data
    net_input = {
        "src_tokens": [],
        "src_coord": [],
        "src_distance": [],
        "src_edge_type": [],
    }
    target = {"finetune_target": []}
    smi_name = []

    # Loop over all data points in the batch
    for item in batch:
        for key in net_input:
            net_input[key].append(tensor(item[f"net_input.{key}"]))
        target["finetune_target"].append(tensor(item["target.finetune_target"]))
        smi_name.append(item["smi_name"])

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
    target["finetune_target"] = torch.stack(target["finetune_target"])
    # smi_name = torch.stack(smi_name)
    # smi_name = torch.zeros_like(target)

    # Return the collated batch
    return {"net_input": net_input, "target": target, "smi_name": smi_name}


batch_size = 64

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)
valid_dataloader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)
test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

# %%
# get the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-6)
criterion = nn.MSELoss()


# train
epochs = 60
num_steps = len(train_dataloader) * epochs

lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer,
    power=1,
    total_iters=num_steps,
)
warmup_period = int(num_steps * 0.06)
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)


best_valid_loss = float("inf")
best_valid_r = float("-inf")
model = model.cuda()
scaler = torch.cuda.amp.grad_scaler.GradScaler(init_scale=4, growth_interval=256)

for epoch in range(epochs):
    model.train()
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)
            target = data["target"]["finetune_target"].to(output.device)
            loss = criterion(output, target)
            # ema.update()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        scaler.update()
        if i % 50 == 0:
            print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")

        with warmup_scheduler.dampening():
            if warmup_scheduler.last_step + 1 >= warmup_period:
                lr_scheduler.step()

    model.eval()

    eval_target = []
    eval_pred = []

    with torch.no_grad():
        total_loss = 0
        for i, data in enumerate(valid_dataloader):
            output = model(data)
            target = data["target"]["finetune_target"].to(output.device)
            eval_target = eval_target + target.cpu().numpy().tolist()
            eval_pred = eval_pred + output.cpu().numpy().tolist()
            loss = criterion(output, target)
            # rmse
            total_loss += np.sqrt(loss.item())
        print(f"Epoch {epoch}, Valid Loss {total_loss/len(valid_dataloader)}")
        eval_target = np.array(eval_target).squeeze()
        eval_pred = np.array(eval_pred).squeeze()
        eval_r = pearsonr(eval_target, eval_pred).statistic
        if eval_r > best_valid_r:
            best_valid_r = eval_r
            print(
                f"New Best Model; Saving..; Loss {total_loss/len(valid_dataloader)}; R {eval_r}"
            )
            # print(f"New Best Model; Saving..; R {best_valid_r}")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/model_weight/unimol_best_finetuned.pt",
            )

# test
torch.save(
    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/model_weight/unimol_last_finetuned.pt",
)

# load best model
model_weight = torch.load(
    "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/model_weight/unimol_best_finetuned.pt"
)["model"]
model.load_state_dict(model_weight)
model.eval()
preds = []
targets = []
smi = []
with torch.no_grad():
    total_loss = 0
    for i, data in enumerate(test_dataloader):
        output = model(data)
        target = data["target"]["finetune_target"].to(output.device)
        loss = criterion(output, target)
        total_loss += loss.item()
        preds.append(output.cpu().numpy().squeeze())
        targets.append(target.cpu().numpy().squeeze())
        smi.append(data["smi_name"])
    print(f"Test Loss {total_loss/len(test_dataloader)}")
    print("Done")

res_dict = {
    "preds": np.concatenate(preds),
    "targets": np.concatenate(targets),
    "smiles": np.concatenate(smi),
}

df = pd.DataFrame(res_dict)
df = df.groupby("smiles").mean().reset_index()


print("best model r:", pearsonr(df["preds"], df["targets"]))


save_dir = "/home/pangkuan/dev/SDL-LNP/model/evaluation/notebooks/prediction"

os.makedirs(save_dir, exist_ok=True)
df.to_csv(os.path.join(save_dir, "uni-mol.csv"))
