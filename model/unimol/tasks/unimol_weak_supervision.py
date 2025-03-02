# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    RawArrayDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    Add2DConformerDataset,
)
from unimol.data.add_2d_conformer_dataset import smi2_2Dcoords
from unicore.tasks import UnicoreTask, register_task
import lmdb
import pickle
from functools import lru_cache
import pandas as pd
from unicore.data import BaseWrapperDataset


logger = logging.getLogger(__name__)



class Add2DConformerAndPropertyDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates, property_target):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.property_target = property_target
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        assert len(atoms) > 0
        smi = self.dataset[index][self.smi]
        property_target = self.dataset[index][self.property_target]
        coordinates_2d = smi2_2Dcoords(smi)
        coordinates = self.dataset[index][self.coordinates]
        coordinates.append(coordinates_2d)
        return {"smi": smi, 
                "atoms": atoms, 
                "coordinates": coordinates,
                "property_target": property_target}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class NestedKeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, nested_key, key):
        self.dataset = dataset
        self.nested_key = nested_key
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.nested_key not in self.dataset[idx] or self.key not in self.dataset[idx][self.nested_key]:
            return torch.tensor(float('nan'))
        return torch.tensor(self.dataset[idx][self.nested_key][self.key])



class MergedLMDBDatasetForPrediction:
    def __init__(self, db_path_list, 
                 prediction_csv_path,
                 smi2idx_mapping,
                 target_label = ("most_apka", "most_bpka", "logp", "logd")):
        self.db_path_list = db_path_list
        self.target_label = target_label
        for db_path in self.db_path_list:
            assert os.path.exists(db_path), f"LMDB file not found: {db_path}"
                
        self.env_list = [self.connect_db(db_path) for db_path in self.db_path_list]
        self._keys_list = []
        for env in self.env_list:
            with env.begin() as txn:
                self._keys_list.append(list(txn.cursor().iternext(values=False)))
        self.cumulative_length = [0]
        for keys in self._keys_list:
            self.cumulative_length.append(len(keys) + self.cumulative_length[-1])
        self.prediction_csv = pd.read_csv(prediction_csv_path)
        self.smi2idx_mapping = smi2idx_mapping
        
        self.lmdb_part_len = sum([len(keys) for keys in self._keys_list])
        self.property_len = len(self.prediction_csv)

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
        return self.lmdb_part_len + self.property_len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        target_property_dict = {}
        if idx < self.lmdb_part_len:
            if not hasattr(self, "env_list"):
                self.env_list = [self.connect_db(db_path) for db_path in self.db_path_list]
            db_index = next(i for i, cl in enumerate(self.cumulative_length) if cl > idx) - 1
            relative_idx = idx - self.cumulative_length[db_index]
            key = self._keys_list[db_index][relative_idx]
            with self.env_list[db_index].begin() as txn:
                item = txn.get(key)
            data = pickle.loads(item)
        else:
            smi = self.prediction_csv.iloc[idx - self.lmdb_part_len]["smiles"].replace("\n", "")
            target_property_dict = self.prediction_csv.iloc[idx - self.lmdb_part_len][self.target_label].to_dict()
            if smi not in self.smi2idx_mapping:
                # print(f"SMILES {smi} not found in smi2idx mapping")
                return self.__getitem__(idx - 1)
            data_path, relative_idx = self.smi2idx_mapping[smi]
            db_index = self.db_path_list.index(data_path)
            key = self._keys_list[db_index][relative_idx]
            with self.env_list[db_index].begin() as txn:
                item = txn.get(key)
            data = pickle.loads(item)
        
        data["property_target"] = target_property_dict
        return data



def list_of_strings(arg):
    return arg.split(',')


@register_task("unimol_weak_supervision")
class UniMolWeakSupervisionTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        type=list_of_strings)
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )
        parser.add_argument(
            "--property-prediction-csv",
            default="/scratch/ssd004/datasets/cellxgene/3d_molecule_data/chembl",
            type=str,
            help="path to the directory containing property prediction csv file",
        )
        parser.add_argument(
            "--smi-idx-mapping",
            default="/scratch/ssd004/datasets/cellxgene/3d_molecule_data/weak-supervision",
            type=str,
            help="path to the directory containing smi to idx mapping",
        )
        parser.add_argument(
            "--target-label",
            default="most_apka,most_bpka,logp,logd",
            type=list_of_strings,
            help="target label for property prediction",
        )
        parser.add_argument(
            "--weak-sup-loss",
            default=1.0,
            type=float,
            help="weak supervision loss weight",
        )
            
            

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        self.smi2idx_mapping  = {}
        

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data[0], args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)
    
    def load_mapping(self, split):
        with open(os.path.join(self.args.smi_idx_mapping, f"{split}.pkl"), "rb") as f:
            logger.info(f"Loading {split} smi2idx mapping from {os.path.join(self.args.smi_idx_mapping, f'{split}.pkl')}")
            self.smi2idx_mapping[split] = pickle.load(f)
        

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        self.load_mapping(split)
        
        split_path_list = []
        for data_path in self.args.data:
            if not os.path.exists(os.path.join(data_path, split + ".lmdb")):
                logger.info(f"LMDB file not found: {os.path.join(data_path, split + '.lmdb')}, skipping")
                continue
            split_path_list.append(os.path.join(data_path, split + ".lmdb"))

        raw_dataset = MergedLMDBDatasetForPrediction(split_path_list,
                                                     os.path.join(self.args.property_prediction_csv, f"{split}.csv"),
                                                     self.smi2idx_mapping[split],
                                                     target_label = self.args.target_label)

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            if self.args.mode =='train':
                raw_dataset = Add2DConformerAndPropertyDataset(
                    raw_dataset, "smi", "atoms", "coordinates", "property_target"
                )
            dataset_collection ={key: NestedKeyDataset(raw_dataset, "property_target", key) for key in self.args.target_label}
            smi_dataset = KeyDataset(raw_dataset, "smi")
            dataset = ConformerSampleDataset(
                raw_dataset, coord_seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(raw_dataset, dataset)
            dataset = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            )
            dataset = CroppingDataset(
                dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
            )
            dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = TokenizeDataset(
                token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(dataset, "coordinates")
            expand_dataset = MaskPointsDataset(
                token_dataset,
                coord_dataset,
                self.dictionary,
                pad_idx=self.dictionary.pad(),
                mask_idx=self.mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=mask_seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
            )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_target_dataset = KeyDataset(expand_dataset, "targets")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )
            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)
            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    encoder_distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            }, {
                "tokens_target": RightPadDataset(
                    tgt_dataset, pad_idx=self.dictionary.pad()
                ),
                "distance_target": RightPadDataset2D(distance_dataset, pad_idx=0),
                "coord_target": RightPadDatasetCoord(coord_dataset, pad_idx=0),
                "smi_name": RawArrayDataset(smi_dataset),
                **dataset_collection
            }

        net_input, target = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model

    def disable_shuffling(self) -> bool:
        return True
    
