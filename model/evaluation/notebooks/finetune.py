# %% [markdown]
# # Finetuning after getting an iteration of experiment results

import argparse
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Finetune the model with k-fold cross validation. First step, generating a new lmdb dataset containing the lipids and results of the previous iterations"
)
parser.add_argument(
    "-f",
    "--file",
    required=True,
    type=str,
    help="Path to the experiment result file, e.g. exp1001_export.csv",
)
args = parser.parse_args()
print(args)


# %%
class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        self.env = self.connect_db(self.db_path)
        with self.env.begin() as txn:
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
        datapoint_pickled = self.env.begin().get(bytes(str(idx), "utf-8"))
        data = pickle.loads(datapoint_pickled)
        return data


# %% [markdown]
# ### 1. making new lmdb dataset containing the lipids and results of the previous iterations
#
# This can takes a file of such result data, that usually can be obtained from the dashboard visualize_result page
#
# The format of the lmdb to be created can be found in `model/pretrain/notebooks/test_lmdb.ipynb`. Essentially, every entry should contains the following fields: 'atoms', 'coordinates', 'mol', 'smi', 'target'
#

# %%
exp_res_file = Path(args.file)
if os.environ["USER"] == "sdl":
    # on the sdl workstation
    library_folder = Path("/home/sdl/vector_cellxgene_data/220k-lib/lmdb")
else:
    library_folder = Path("/datasets/cellxgene/3d_molecule_data/220k-lib/lmdb")
# the 220k lmdbs are split into 10 subfolders library_folder/0, library_folder/1, ..., library_folder/9
library_lmdb_paths = [library_folder / str(i) / "test.lmdb" for i in range(10)]
library_smi_paths = [library_folder / str(i) / "smi_name_list.txt" for i in range(10)]
assert all([p.exists() for p in library_lmdb_paths])
assert all([p.exists() for p in library_smi_paths])

exp_res_df = pd.read_csv(exp_res_file, index_col=0)
# shuffle the data rows
idx = np.random.permutation(exp_res_df.index)
exp_res_df = exp_res_df.loc[idx]
assert exp_res_df["max"].is_monotonic_increasing is False
assert exp_res_df["max"].is_monotonic_decreasing is False
exp_res = [
    {
        "smiles": row["smiles"],
        "target": row["max"],
    }
    for _, row in exp_res_df.iterrows()
]
print(f"Total {len(exp_res)} experiment results loaded from {exp_res_file}")

# load the library lmdbs
library_datasets = [LMDBDataset(str(p)) for p in library_lmdb_paths]

# make smiles to idx mapping, 10 mappings for 10 subfolders
smi_to_idx_mappings = []
for smi_path in library_smi_paths:
    with open(smi_path, "r") as f:
        smi_to_idx = {smi.strip(): idx for idx, smi in enumerate(f)}
    smi_to_idx_mappings.append(smi_to_idx)
print(f"number of smiles in mappings: {sum([len(m) for m in smi_to_idx_mappings])}")


# %%
# find the folder num and idx in the folder for each exp_res
# and get the corresponding molecule info
for exp_res_item in exp_res:
    smi = exp_res_item["smiles"]
    for folder_num, smi_to_idx in enumerate(smi_to_idx_mappings):
        if smi in smi_to_idx:
            idx = smi_to_idx[smi]
            break
    else:
        raise ValueError(f"smiles {smi} not found in any of the mappings")
    dataset = library_datasets[folder_num]
    data = dataset[idx]
    assert (
        data["smi"].strip() == smi
    ), f"folder {folder_num} idx {idx}: {data['smi']} != {smi}"
    exp_res_item["folder_num"] = folder_num
    exp_res_item["idx"] = idx
    exp_res_item["atoms"] = data["atoms"]
    exp_res_item["coordinates"] = data["coordinates"]
    # exp_res_item["mol"] = data["mol"]

# %%
exp_res_to_save = [
    {
        "atoms": item["atoms"],
        "coordinates": item["coordinates"],
        # "mol": item["mol"],
        "smi": item["smiles"],
        "target": item["target"],
    }
    for item in exp_res
]

# save the results to new lmdb
output_lmdb_path = exp_res_file.with_suffix(".lmdb")
if output_lmdb_path.exists():
    raise ValueError(f"{output_lmdb_path} already exists")
env = lmdb.open(
    str(output_lmdb_path),
    subdir=False,
    map_size=1099511627776 * 2,
    readonly=False,
    meminit=False,
    map_async=True,
    max_dbs=0,
    lock=False,
    max_readers=1,
)
with env.begin(write=True) as txn:
    for idx, exp_res_item in enumerate(exp_res_to_save):
        txn.put(str(idx).encode(), pickle.dumps(exp_res_item))
print(f"results saved to {output_lmdb_path}")
