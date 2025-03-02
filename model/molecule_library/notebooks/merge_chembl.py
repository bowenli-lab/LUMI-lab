import lmdb
import os
import shutil



def merge_lmdbs(input_dir, output_name):
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(500e10),
    )
    txn_write = env_new.begin(write=True)

    def get_lmdb_files(directories):
        lmdb_files = []
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                if any(file.endswith('.lmdb') for file in files):
                    lmdb_files.append(root)
        return lmdb_files

    def read_and_write_data(lmdb_dirs):
        count = 0
        for lmdb_dir in lmdb_dirs:
            print(f"processing: {lmdb_dir}; starting")
            temp_lmdb_dir = lmdb_dir
            temp_lmdb_file = temp_lmdb_dir + "/map.lmdb"
            current_file_count = 0
            env_read = lmdb.open(temp_lmdb_file, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
            with env_read.begin(write=False) as txn_read:
                for key, value in txn_read.cursor():
                    txn_write.put(str(count).encode("ascii"), value)
                    count += 1
                    current_file_count += 1
            print(f"finished: {lmdb_dir}; items {current_file_count}")
        return count

    lmdb_files = get_lmdb_files(input_dir)
    total_written = read_and_write_data(lmdb_files)

    txn_write.commit()
    env_new.close()

    print(f"Total entries written: {total_written}")

# make it for train and valid

train_val_dict = {
    "train": '/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/conformation/missing_conformer',
}

for key, value in train_val_dict.items():
    print(f"processing {key}: {value}")


    input_directory = [
                        value,
                    ]

    dst = f"/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl/conformation/{key}.lmdb"


    print(f"outputing data to local disk: {dst}")


    merge_lmdbs(input_directory, dst, )


