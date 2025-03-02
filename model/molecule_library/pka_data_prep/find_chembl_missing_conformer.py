# %% [markdown]
# We search the conformers from existing data first, if not found, we generate the conformers by RDKit.

# %%
from unicore.data import LMDBDataset
import pandas as pd
import concurrent.futures

train_data_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/cleaned_ligands/train.lmdb"
valid_data_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/cleaned_ligands/valid.lmdb"

num_workers = 12

def hash_smiles(smiles):
    # return hashlib.sha256(smiles.encode('utf-8')).hexdigest()
    return smiles # identity projection

# Assuming dataset is already loaded as LMDBDataset
dataset = LMDBDataset(train_data_path)
print(f"Loaded {len(dataset)} rows from lmdb training set")

# Function to process a chunk of data
def process_chunk(chunk):
    local_hash_to_indices = {}
    for i in chunk:
        hash_value = hash_smiles(dataset[i]["smi"])
        if hash_value not in local_hash_to_indices:
            local_hash_to_indices[hash_value] = []
        local_hash_to_indices[hash_value].append(i)
    return local_hash_to_indices

N = len(dataset)
chunk_size = (N + num_workers - 1) // num_workers  # Ensure all data is covered

# Split indices into chunks
chunks = [range(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

# Process each chunk in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(process_chunk, chunks))

# Combine results
hash_to_indices = {}
for result in results:
    for key, value in result.items():
        if key not in hash_to_indices:
            hash_to_indices[key] = value
        else:
            hash_to_indices[key].extend(value)

print(f"Found {len(hash_to_indices)} unique hashes in the dataset")

# Process pka data
pka_csv_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/original_data/chembl_34_pka.csv"
df = pd.read_csv(pka_csv_path)
print(f"Loaded {len(df)} rows from chembl dataset")

def process_pka_row(i):
    smiles = df["smiles"][i]
    hash_value = hash_smiles(df["smiles"][i])
    if hash_value not in hash_to_indices:
        return smiles
    else: 
        return None
    
with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(process_pka_row, range(len(df))))
    
# return the non-None list
missing_smiles = [x for x in results if x is not None]

print(f"Found {len(missing_smiles)} missing smiles in the dataset")

with open("/h/pangkuan/dev/SDL-LNP/model/molecule_library/pka_data_prep/missing_chembl_conformer.txt", "w") as f:
    for item in missing_smiles:
        f.write("%s\n" % item)