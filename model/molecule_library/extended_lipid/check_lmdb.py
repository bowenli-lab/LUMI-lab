import lmdb
import pickle

output_name = "/scratch/ssd004/datasets/cellxgene/3d_molecule_data/cleaned_ligands/train.lmdb"
env_new = lmdb.open(
    output_name,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(300e9),
)
# get first key
with env_new.begin() as txn:
    idx = 10000
    datapoint_pickled = env_new.begin().get(f"{idx}".encode("ascii"))
    data = pickle.loads(datapoint_pickled)
    print(data.keys())

env_new.close()