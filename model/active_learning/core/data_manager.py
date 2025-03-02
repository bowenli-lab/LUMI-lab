# Modified from https://github.com/RelationRx/pyrelational/blob/main/pyrelational/data_managers/data_manager.py


import random
import warnings
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor
from datasets import Dataset
from transformers.data.data_collator import DataCollator
from torch.utils.data import DataLoader



class DataManager:
    """
    DataManager for active learning pipelines, tailored for molecular dataset.

    This class is modified to support huggingface dataset indexed with conformers. 
    
    We maintain two series of indices, molecular indices and conformer indices.
    
    The conformer indices will be used to index the dataset, while the molecular indices 
    will be used to index the labels. There is a many-to-one relationship between molecular
    indices and conformer indices.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_conformers: int = 11,
        label_attr: Optional[str] = None,
        train_indices: Optional[List[int]] = None,
        labelled_indices: Optional[List[int]] = None,
        unlabelled_indices: Optional[List[int]] = None,
        validation_indices: Optional[List[int]] = None,
        test_indices: Optional[List[int]] = None,
        random_label_size: Union[float, int] = 0.1,
        hit_ratio_at: Optional[Union[int, float]] = None,
        random_seed: int = 1234,
    ):
        """
        :param dataset: A huggingface dataset.
        :param label_attr: string indicating name of attribute in the dataset class that correspond to the tensor
            containing the labels/values to be predicted; by default, pyrelational assumes it correspond to dataset.
        :param train_indices: An iterable of indices mapping to training sample indices in the dataset
        :param labelled_indices: An iterable of indices  mapping to labelled training samples
        :param unlabelled_indices: An iterable of indices to unlabelled observations in the dataset
        :param validation_indices: An iterable of indices to observations used for model validation
        :param test_indices: An iterable of indices to observations in the input dataset used for
            test performance of the model
        :param random_label_size: Only used when labelled and unlabelled indices are not provided. Sets the size of
            labelled set (should either be the number of samples or ratio w.r.t. train set)
        :param hit_ratio_at: optional argument setting the top percentage threshold to compute hit ratio metric
        :param random_seed: random seed used to generate labelled/unlabelled splits when none are provided.
        """


        self.dataset = dataset
        self.label_attr = label_attr
        self.index2label_dict = {} #conformer idx
        if label_attr is not None:
            # molecule's index
            for i in range(len(dataset)):
                self.index2label_dict[i] = dataset[i][label_attr]
                
        # resolve molecule2conformer index mapping
        self.num_conformers = num_conformers
        self.molecule_num = len(dataset) // num_conformers
        self.molecule2conformer = {i: [i*num_conformers + j for j in range(num_conformers)] for i in range(self.molecule_num)}
        self.conformer2molecule = {j: i for i in self.molecule2conformer for j in self.molecule2conformer[i]}

        # Resolve masks and the values they should take given inputs
        self._resolve_dataset_split_indices(train_indices, validation_indices, test_indices)

        # Set l and u indices according to mask arguments
        # and need to check that they aren't part of
        if labelled_indices is not None:
            if unlabelled_indices is not None:
                self._ensure_no_l_u_intersection(labelled_indices, unlabelled_indices)
            else:
                unlabelled_indices = list(set(self.train_indices) - set(labelled_indices))
            self.labelled_indices = labelled_indices
            self.l_indices = labelled_indices
            self.unlabelled_indices = unlabelled_indices
            self.u_indices = unlabelled_indices
        else:
            print("## Labelled and/or unlabelled mask unspecified")
            self.random_label_size = random_label_size
            self._generate_random_initial_state(random_seed)
        self._ensure_no_l_or_u_leaks()
        self._ensure_unique_indices()
        self._top_unlabelled_set(hit_ratio_at)
        
    def _get_conformer_index_list(self, molecule_list: List[int]) -> List[int]:
        """
        Get conformer indices from a list of molecule indices.

        :param molecule_list: list of molecule indices
        :return: list of conformer indices
        """
        conformer_list = []
        for i in molecule_list:
            conformer_list.extend(self.molecule2conformer[i])
        return conformer_list

    @staticmethod
    def _ensure_no_split_leaks(
        train_indices: List[int],
        validation_indices: Optional[List[int]],
        test_indices: List[int],
    ) -> None:
        """Ensures that there is no overlap between train/validation/test sets."""
        tt = set.intersection(set(train_indices), set(test_indices))
        tv, vt = None, None
        if validation_indices is not None:
            tv = set.intersection(set(train_indices), set(validation_indices))
            vt = set.intersection(set(validation_indices), set(test_indices))
        if tv or tt or vt:
            raise ValueError("There is an overlap between the split indices supplied")

    def _ensure_unique_indices(self) -> None:
        """
        Makes sure that all the indices have no repeated values,
        and that the train indices are the union of the labelled and unlabelled indices.
        raises a ValueError if any of these conditions are not met.
        """
        if len(set(self.l_indices)) != len(self.l_indices):
            raise ValueError("There are repeated values in labelled indices")
        if len(set(self.u_indices)) != len(self.u_indices):
            raise ValueError("There are repeated values in unlabelled indices")
        if self.validation_indices is not None and len(set(self.validation_indices)) != len(self.validation_indices):
            raise ValueError("There are repeated values in validation indices")
        if len(set(self.test_indices)) != len(self.test_indices):
            raise ValueError("There are repeated values in test indices")

        # Check that the train indices are the union of the labelled and unlabelled indices
        if set(self.train_indices) != set(self.l_indices + self.u_indices):
            raise ValueError("The train indices are not the union of the labelled and unlabelled indices")

    @staticmethod
    def _ensure_not_empty(mode: Literal["train", "test"], indices: List[int]) -> None:
        """
        Ensures that train or test set is not empty.

        :param mode: either "train" or "test"
        :param indices: either train or test indices
        """
        if len(indices) == 0:
            raise ValueError(f"The {mode} set is empty")

    @staticmethod
    def _ensure_no_l_u_intersection(labelled_indices: List[int], unlabelled_indices: List[int]) -> None:
        """ "
        Ensure that there is no overlap between labelled and unlabelled samples.

        :param labelled_indices: list of indices in dataset which have been labelled
        :param unlabelled_indices: list of indices in dataset which have not been labelled
        """
        if set.intersection(set(labelled_indices), set(unlabelled_indices)):
            raise ValueError("There is overlap between labelled and unlabelled samples")

    def _ensure_no_l_or_u_leaks(self) -> None:
        """
        Ensures that there are no leaks of labelled or unlabelled indices
        in the validation or tests indices.
        """
        if self.validation_indices is not None:
            v_overlap = set.intersection(set(self.l_indices), set(self.validation_indices))
            if v_overlap:
                raise ValueError(
                    f"There is {len(v_overlap)} sample overlap between the labelled indices and the validation set"
                )
            v_overlap = set.intersection(set(self.u_indices), set(self.validation_indices))
            if v_overlap:
                raise ValueError(
                    f"There is {len(v_overlap)} sample overlap between the unlabelled indices and the validation set"
                )

        if self.test_indices is not None:
            t_overlap = set.intersection(set(self.l_indices), set(self.test_indices))
            if t_overlap:
                raise ValueError(
                    f"There is {len(t_overlap)} sample overlap between the labelled indices and the test set"
                )

            # save memory by using same variables
            t_overlap = set.intersection(set(self.u_indices), set(self.test_indices))
            if t_overlap:
                raise ValueError(
                    f"There is {len(t_overlap)} sample overlap between the unlabelled indices and the test set"
                )

    def _resolve_dataset_split_indices(
        self,
        train_indices: Optional[List[int]],
        validation_indices: Optional[List[int]],
        test_indices: Optional[List[int]],
    ) -> None:
        """
        This function is used to resolve what values the indices should be given
        when only a partial subset of them is supplied


        :param train_indices: list of indices in dataset for train set
        :param validation_indices: list of indices in dataset for validation set
        :param test_indices: list of indices in dataset for test set
        """

        remaining_indices = set(range(len(self.molecule2conformer))) - set.union(
            set(train_indices if train_indices is not None else []),
            set(validation_indices if validation_indices is not None else []),
            set(test_indices if test_indices is not None else []),
        )

        if train_indices is None:
            if test_indices is None:
                raise ValueError("No train or test specified, too ambiguous to set values")
            train_indices = list(remaining_indices)
        elif test_indices is None:
            test_indices = list(remaining_indices)
        elif remaining_indices:
            warnings.warn(f"{len(remaining_indices)} indices are not found in any split", stacklevel=3)

        self._ensure_not_empty("train", train_indices)
        self._ensure_not_empty("test", test_indices)
        self._ensure_no_split_leaks(train_indices, validation_indices, test_indices)
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.test_indices = test_indices

    def __len__(self) -> int:
        # Override this if necessary
        return len(self.molecule2conformer)

    def __getitem__(self, idx: int):
        """
        Access samples by index directly.
        This samples the dataset by molecular index.
        """
        molecular_idx = self.molecule2conformer[idx]
        return self.dataset[molecular_idx]

    def set_target_value(self, idx: int, value: Any) -> None:
        """
        Sets a value to the label of the observation at the given molecular index.

        :param idx: index value to the observation
        :param value: new value for the observation
        """
        conformer_idx = self.molecule2conformer[idx]
        for i in conformer_idx:
            self.index2label_dict[i] = value

    def _top_unlabelled_set(self, percentage: Optional[Union[int, float]] = None) -> None:
        """
        Sets the top unlabelled indices according to the value of their labels.
        Used for calculating hit ratio, which demonstrates
        how quickly the samples in this set are recovered for labelling.

        :param percentage: Top percentage of samples to be considered in top set
        """
        if percentage is None:
            self.top_unlabelled = None
        else:
            if isinstance(percentage, int):
                percentage /= 100
            assert 0 < percentage < 1, "hit ratio's percentage should be strictly between 0 and 1 (or 0 and 100)"
            ixs = self.u_indices
            percentage = int(percentage * len(ixs))
            y = self.get_sample_labels(ixs)
            threshold = np.sort(y.abs())[-percentage]
            indices = torch.where(y.abs() >= threshold)[0]
            self.top_unlabelled = set(ixs[i] for i in indices)

    def get_train_set(self) :
        """Get train set from full dataset and train indices."""
        # TODO: update prediction labels.
        conformers = self._get_conformer_index_list(self.train_indices)
        train_subset = self.dataset.select(conformers)
        return train_subset

    def get_validation_set(self):
        """Get validation set from full dataset and validation indices."""
        if self.validation_indices is None:
            return None
        conformers = self._get_conformer_index_list(self.validation_indices)
        validation_subset = self.dataset.select(conformers)
        return validation_subset

    def get_test_set(self):
        """Get test set from full dataset and test indices."""
        conformers = self._get_conformer_index_list(self.test_indices)
        test_subset = self.dataset.select(conformers)
        return test_subset

    def get_labeled_set(self):
        """Get labelled set from full dataset and labelled indices."""
        conformers = self._get_conformer_index_list(self.l_indices)
        labelled_subset = self.dataset.select(conformers)
        return labelled_subset

    def get_unlabelled_set(self):
        """Get unlabelled set from full dataset and unlabelled indices."""
        conformers = self._get_conformer_index_list(self.u_indices)
        unlabelled_subset = self.dataset.select(conformers)
        return unlabelled_subset


    def _generate_random_initial_state(self, seed: int = 0) -> None:
        """
        Process the dataset to produce a random subsets of labelled and unlabelled
        samples from the dataset based on the ratio given at initialisation and creates
        the data_loaders

        :param seed: random seed for reproducibility
        """
        if isinstance(self.random_label_size, float):
            assert 0 < self.random_label_size < 1, "if a float, random_label_size should be between 0 and 1"
            num_labelled = int(self.random_label_size * len(self.train_indices))
        else:
            num_labelled = self.random_label_size

        print("## Randomly generating labelled subset with {} samples from the train data".format(num_labelled))
        random.seed(seed)
        l_indices = set(random.sample(self.train_indices, num_labelled))
        u_indices = set(self.train_indices) - set(l_indices)

        self.l_indices = list(l_indices)
        self.u_indices = list(u_indices)

    def update_train_labels(self, indices: List[int]) -> None:
        """
        Updates the labelled and unlabelled sets of the dataset.

        Different behaviour based on whether this is done in evaluation mode or real mode.
        The difference is that in evaluation mode the dataset already has the label, so it
        is a matter of making sure the observations are moved from the unlabelled set to the
        labelled set.

        :param indices: list of indices corresponding to samples which have been labelled
        """
        self.l_indices = list(set(self.l_indices + indices))
        self.u_indices = list(set(self.u_indices) - set(indices))

    def get_percentage_labelled(self) -> float:
        """
        Percentage of total available dataset labelled.

        :return: percentage value
        """
        total_len = len(self.l_indices) + len(self.u_indices)
        num_labelled = len(self.l_indices)
        return (num_labelled / float(total_len)) * 100

    def get_sample_feature_vector(self, ds_index: int) -> Any:
        """To be reviewed for deprecation (for datasets without tensors)"""
        sample = self[ds_index]
        ret = sample[0].flatten()
        return ret

    def get_sample_feature_vectors(self, ds_indices: List[int]) -> List[Tensor]:
        """To be reviewed for deprecation (for datasets without tensors)"""
        res = []
        for ds_index in ds_indices:
            res.append(self.get_sample_feature_vector(ds_index))
        return res

    def get_sample_labels(self, ds_indices: List[int]) -> Tensor:
        """
        Get sample labels. This assumes that labels are last element in output of dataset

        :param ds_indices: collection of indices for accessing samples in dataset.
        :return: list of labels for provided indexes
        """
        res = []
        for ds_index in ds_indices:
            label = self.index2label_dict[ds_index]
            res.append(label)
        return torch.stack(res)



    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        """Pretty print a summary of the data_manager contents"""
        str_percentage_labelled = "%.3f" % (self.get_percentage_labelled())
        str_out = self.__repr__()
        if self.train_indices is not None:
            str_out += "\nTraining set size: {} molecules ({} conformers).\n".format(len(self.train_indices), len(self.train_indices)*self.num_conformers)
        if self.l_indices is not None:
            str_out += "Labelled: {}, Unlabelled: {}\n".format(len(self.l_indices), len(self.u_indices))
        str_out += "Percentage Labelled: {}".format(str_percentage_labelled)

        return str_out