import sys
from pathlib import Path
sys.path.insert(0, "../../")
from unimol.hf_unimol import UniMol, UniMolConfig, init_unimol_backbone
from datasets import Dataset
from unimol.lmdb_dataset import collate_fn, load_dataset, SliceDataset

backbone_weight_path ="/home/sdl/3d_molecule_save/pretrain-20240602-1143/checkpoint_best.pt"
dict_path = "/home/sdl/SDL-LNP/model/serverless/docker/serverless-infer/dict.txt"
lmdb_dir = Path("/home/sdl/SDL-LNP/model/serverless/docker/serverless-infer/all-lmdb")
_, dictionary = init_unimol_backbone(backbone_weight_path, dict_path=dict_path)
test_data = load_dataset(dictionary, str(lmdb_dir / "test.lmdb"), "test")


smi_name_list = []
for i in test_data:
    smi_name_list.append(i["smi_string"])
    
print(f"len(smi_name_list): {len(smi_name_list)}")

with open("smi_name.txt", "w") as f:
    f.write("\n".join(smi_name_list))


