import pandas as pd
from multiprocessing import Pool, cpu_count
import os

filename = '/scratch/ssd004/datasets/cellxgene/3d_molecule_data/15M_virtual_library.csv'
chunksize = 100000  # Adjust the chunk size according to your memory capacity

def find_duplicates(chunk):
    # Return the duplicated `combined_mol_SMILES` in the chunk
    return chunk[chunk.duplicated('combined_mol_SMILES', keep=False)]

def process_file_in_chunks(filename, chunksize):
    reader = pd.read_csv(filename, chunksize=chunksize)
    with Pool(cpu_count()) as pool:
        duplicates = pool.map(find_duplicates, reader)
    return pd.concat(duplicates)

if __name__ == '__main__':
    duplicates_df = process_file_in_chunks(filename, chunksize)
    duplicates_df.to_csv('duplicated_combined_mol_SMILES.csv', index=False)
    print(f"Found {len(duplicates_df)} duplicated entries.")

# Run the script
# os.system("/h/pangkuan/miniconda3/envs/unimol/bin/python check_dup.py")
