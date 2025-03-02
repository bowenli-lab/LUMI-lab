from unicore.data import LMDBDataset
import concurrent.futures
import os
import lmdb
import pickle

train_data_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/ligands/train.lmdb"
valid_data_path = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/ligands/valid.lmdb"

OUTPUT_PATH = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/cleaned_ligands"

num_workers = os.cpu_count()

def hash_smiles(smiles):
    # return hashlib.sha256(smiles.encode('utf-8')).hexdigest()
    return smiles # identity projection
 


for partition, data_path in {"train": train_data_path, "valid": valid_data_path}.items():
    # Assuming dataset is already loaded as LMDBDataset
    dataset = LMDBDataset(data_path)
    print(f"Loaded {len(dataset)} rows from lmdb {partition} set")

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

    # select only non-duplicated indices to save
    non_duplicate_indices = []
    for key, value in hash_to_indices.items():
        non_duplicate_indices.append(value[0])
    
    output_name = os.path.join(OUTPUT_PATH, f"{partition}.lmdb")
    # Save non-duplicated indices to a new lmdb
    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(300e9),
    )
    txn_write = env_new.begin(write=True)
    i = 0
    for idx in non_duplicate_indices:
        entry = dataset[idx]
        if entry is not None:
            txn_write.put(f'{i}'.encode("ascii"), 
                          pickle.dumps(entry, protocol=-1))
            i += 1
    print('Process {} lines for {}'.format(i, partition))
    txn_write.commit()
    env_new.close()
    

