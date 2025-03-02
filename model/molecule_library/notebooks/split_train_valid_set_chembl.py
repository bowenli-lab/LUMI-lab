import pandas
from unicore.data import LMDBDataset

# 1. find all entries that are in valid but not in train
# 2. produce a new valid and train spread sheet that has no overlap

FULL_CSV_PATH = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/original_data/chembl_34_molecular_property.csv"
TRAIN_CSV_PATH = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/conformation/train.csv"
VALID_CSV_PATH = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/conformation/valid.csv"

VALID_LMDB_PATH = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/cleaned_ligands/valid.lmdb"

# load data
full_data = pandas.read_csv(FULL_CSV_PATH)
print(f"Loaded {len(full_data)} rows from {FULL_CSV_PATH}")

# get all smiles in this lmdb
valid_dataset = LMDBDataset(VALID_LMDB_PATH)
valid_smiles = set([row["smi"] for row in valid_dataset])

# get the overlap between valid and full
overlap = valid_smiles.intersection(set(full_data["smiles"]))
print(f"Overlap: {len(overlap)}")

# save it to a new csv
valid_data = full_data[full_data["smiles"].isin(overlap)]
train_data = full_data[~full_data["smiles"].isin(overlap)]

# valid_data.to_csv(VALID_CSV_PATH, index=False)
# train_data.to_csv(TRAIN_CSV_PATH, index=False)