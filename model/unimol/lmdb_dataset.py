import os
import numpy as np
import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from unicore.data import (
    AppendTokenDataset,
    Dictionary,
    FromNumpyDataset,
    LMDBDataset,
    NestedDictionaryDataset,
    PrependTokenDataset,
    RawArrayDataset,
    RawLabelDataset,
    RightPadDataset,
    RightPadDataset2D,
    SortDataset,
    TokenizeDataset,
    UnicoreDataset,
)

from .data import (
    AtomTypeDataset,
    ConformerSampleDataset,
    CroppingDataset,
    DistanceDataset,
    EdgeTypeDataset,
    KeyDataset,
    NormalizeDataset,
    RemoveHydrogenDataset,
    RightPadDatasetCoord,
    data_utils,
)
from .data.tta_dataset import TTADataset
from .models import UniMolModel, unimol_base_architecture

NUM_CONFORMER = 11


def myHash(text:str):
    text = str(text)
    hash=0
    for ch in text:
        hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
    return int(hash)


def collate_fn(batch):
    # Initialize dictionaries to hold processed batch data
    net_input = {
        "src_tokens": [],
        "src_coord": [],
        "src_distance": [],
        "src_edge_type": [],
    }
    target = {"target": []}
    smi_name = []

    # Loop over all data points in the batch
    for item in batch:
        for key in net_input:
            net_input[key].append(torch.tensor(item[f"{key}"]))
        target["target"].append(torch.tensor(item["target"]))
        smi_name.append(torch.tensor(item["smi_name"]))

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
    target = torch.stack(target["target"])
    smi_name = torch.stack(smi_name)    

    # Return the collated batch
    return {"target": target, "smi_name": smi_name, 
            **net_input}


class HashArrayDataset(UnicoreDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        output = myHash(str(self.dataset[index]))
        return output

    def __len__(self):
        return len(self.dataset)


    def collater(self, samples):
        return default_collate(samples)
    

def compute_hash2smi(dataset,):
    hash2smi = {}
    for i in range(len(dataset)):
        smi_string = dataset[i]["smi_string"]
        hash2smi[myHash(smi_string)] = smi_string
    return hash2smi



class BinaryTensorDataset(UnicoreDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def _encode_str2bin(self, string):
        byte_lst = list(bytes(string, 'utf8'))
        return np.array(byte_lst, dtype=np.uint8)

    def __getitem__(self, index):
        return_data =  self._encode_str2bin(self.dataset[index])
        return return_data

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return default_collate(samples)


class StringKeyDataset(KeyDataset):
    def __getitem__(self, idx):
        return str(self.dataset[idx][self.key])

class ComputeMolecularDescriptorDataset(KeyDataset):
    
    def __init__(self, dataset, key):
        super().__init__(dataset, key)
        self._cache = {}
    
    def _cache_smi(self, smiles):
        if smiles not in self._cache:
            self._cache[smiles] = self._get_mordred_descriptors_from_smiles(smiles)
        return self._cache[smiles]
        
    
    def _get_mordred_descriptors_from_smiles(self, smiles):
        from rdkit import Chem
        from mordred import Calculator, descriptors
        mol = Chem.MolFromSmiles(smiles)
        calc = Calculator(descriptors,
                          ignore_3D=True,)
        result = calc.pandas([mol], 
                             quiet=True,
                             )
        vector = result.values[0].astype(float)
        return vector
    
    def __getitem__(self, idx):
        return self._cache_smi(self.dataset[idx][self.key])

def load_dataset(dictionary, split_path, split, topk_conformer=NUM_CONFORMER):
    """Load a given dataset split.
    Args:
        split (str): name of the data scoure (e.g., train)
        hash2smi (dict): hash to smi mapping
        topk_conformer (int): number of conformers to sample
    """
    dataset = LMDBDataset(split_path)
    
    print(f"Loading dataset, selecting top {topk_conformer} conformers.")

    dataset = TTADataset(dataset, 0, "atoms", "coordinates", topk_conformer)
    dataset = AtomTypeDataset(dataset, dataset)
    tgt_dataset = KeyDataset(dataset, "target")
    smi_dataset = KeyDataset(dataset, "smi")
    smi_string_dataset =  StringKeyDataset(dataset, "smi")

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
            "target": RawLabelDataset(tgt_dataset),
            "smi_name": HashArrayDataset(smi_dataset),
            "smi_string": smi_string_dataset
        },
    )
    datasets = nest_dataset
    return datasets


def load_dataset_molecular_descriptor(
    split_path,
    split,
):

    dataset = LMDBDataset(split_path)

    dataset = TTADataset(dataset, 0, "atoms", "coordinates", NUM_CONFORMER)
    dataset = AtomTypeDataset(dataset, dataset)
    tgt_dataset = KeyDataset(dataset, "target")
    smi_dataset = KeyDataset(dataset, "smi")
    smi_string_dataset =  StringKeyDataset(dataset, "smi")
    molecular_descriptor_dataset = ComputeMolecularDescriptorDataset(dataset, "smi")

    nest_dataset = NestedDictionaryDataset(
        {
            "target": RawLabelDataset(tgt_dataset),
            "smi_name": HashArrayDataset(smi_dataset),
            "smi_string": smi_string_dataset,
            "molecular_descriptor": molecular_descriptor_dataset
        },
    )
    datasets = nest_dataset
    return datasets



def collate_fn_molecular_descriptor(batch):
    # Initialize dictionaries to hold processed batch data
    net_input = {
        "molecular_descriptor": [],
    }
    target = {"target": []}
    smi_name = []

    # Loop over all data points in the batch
    for item in batch:
        for key in net_input:
            entry = torch.tensor(item[f"{key}"])
            # fill in nan
            entry[torch.isnan(entry)] = 0
            # log large values
            entry[entry > 1e3] = torch.log(entry[entry > 1e3])
            entry[entry < -1e3] = torch.log(-entry[entry < -1e3])
            # fill in inf
            entry[torch.isinf(entry)] = 0
            # clip large values
            entry = torch.clamp(entry, -1e4, 1e4)
            net_input[key].append(entry)
        target["target"].append(torch.tensor(item["target"]))
        smi_name.append(torch.tensor(item["smi_name"]))

    key = "molecular_descriptor"
    net_input[key] = torch.stack(net_input[key])

    # Convert lists to tensors
    target = torch.stack(target["target"])
    smi_name = torch.stack(smi_name)    

    # Return the collated batch
    return {"target": target, "smi_name": smi_name, 
            **net_input}

class SliceDataset(UnicoreDataset):
    def __init__(self, dataset, start, end):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.end = end
        if self.end > len(self.dataset):
            self.end = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index + self.start]

    def __len__(self):
        return self.end - self.start

