from unicore.data import LMDBDataset
import concurrent.futures
import os
import lmdb
import pickle

OUTPUT_DIR = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/weak-supervision/"


ligand_train_data_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/cleaned_ligands/train.lmdb"
chembl_train_data_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/conformation/train.lmdb"

ligand_valid_data_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/cleaned_ligands/valid.lmdb"


train_data_path_list = [ligand_train_data_path, chembl_train_data_path]
valid_data_path_list = [ligand_valid_data_path]

num_workers = os.cpu_count()

def hash_smiles(smiles):
    # return hashlib.sha256(smiles.encode('utf-8')).hexdigest()
    return smiles # identity projection

SAVE = True

for flag, data_path_list in {
    "train": train_data_path_list,
    "valid": valid_data_path_list
}.items():
    hash_to_indices = {}
    for data_path in data_path_list:
        # Assuming dataset is already loaded as LMDBDataset
        dataset = LMDBDataset(data_path)
        print(f"{flag}: Loaded {len(dataset)} rows from lmdb {data_path} set")

        # Function to process a chunk of data
        def process_chunk(chunk):
            local_hash_to_indices = {}
            for i in chunk:
                hash_value = hash_smiles(dataset[i]["smi"].replace("\n", ""))
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
        for result in results:
            for key, value in result.items():
                if key not in hash_to_indices:
                    hash_to_indices[key] = (data_path, value[0])

        print(f"{flag}: Found {len(hash_to_indices)} unique hashes in the dataset")
        
    output_path = os.path.join(OUTPUT_DIR, f"{flag}.pkl")
    
    if SAVE:
        with open(output_path, "wb") as f:
            pickle.dump(hash_to_indices, f)
    else:
        pass

    print(f"{flag}: Total number of unique smiles: {len(hash_to_indices)}")

    print(f"{flag}: Saved mapping dict to {output_path}")
