from typing import Dict, List, Optional

from bson import ObjectId

from sdl_orchestration.experiment import LipidStructure, Well

from sdl_orchestration.experiment import sample_registry
from sdl_orchestration.experiment.samples.plate96well import Plate96Well


class ReagentMappingHandler:
    """
    This class handles the reagents used in the experiment, resolving and
    providing the necessary info for accessing the reagents.
    """

    def __init__(self, targets: Plate96Well, experiment_id: Optional[ObjectId] = None):
        self.targets = targets
        self.target_lipid_dict = targets.get_lipid_dict()
        self.experiment_id = experiment_id
        self.mapping_head = "targets="

    def resolve_mapping(self) -> str:
        """
        This method resolves the mapping of the reagents to the target lipids.

        Returns:
            str: The mapping of the reagents to the target lipids.
        """
        target2well = self._resolve_target2well()
        dict_string = self._parse_mapping_dict2str(target2well)
        return dict_string

    def resolve_reverse_mapping(self) -> str:
        """
        This method resolves the reverse mapping of the reagents to the target
        lipids.

        Returns:
            str: The reverse mapping of the reagents to the target lipids.
        """
        reverse_mapping = self._resolve_reserved_target2well()
        dict_string = self._parse_mapping_dict2str(reverse_mapping)
        return dict_string

    def _resolve_reserved_target2well(self) -> Dict[int, List[int]]:
        """
        This method resolves the reverse mapping of the reagents to the target
        lipids.

        Note that this method produces a mapping from: source -> [destinations]

        Returns:
            Dict[int, List[int]]: The reverse mapping of the reagents to the target
            lipids.
        """
        target2well = self._resolve_target2well()
        reverse_mapping = {}
        for well_num, well_list in target2well.items():
            for well in well_list:
                if well not in reverse_mapping:
                    reverse_mapping[well] = []
                reverse_mapping[well].append(well_num)
        return reverse_mapping

    def _resolve_target2well(self) -> Dict[int, List[int]]:
        """
        This method resolves the reagent to well mapping, for each target lipid
        given, we resolve the needed wells for each target lipid.

        Under 4CR, each target requires 4 reagents. Each reagent is in a well.


        Note that this method produces a mapping from: dest -> [sources]

        Returns:
            List[List[Well]]: A list of lists of wells, each list of wells
            corresponds to a target lipid.
        """
        res_well_dict = {}
        reagent_plate = sample_registry.reagent
        for well_num, lipid in self.target_lipid_dict.items():
            # resolve the reagent to well mapping
            res_well_dict[well_num] = reagent_plate.map_target2wells(lipid)
        return res_well_dict

    def _parse_mapping_dict2str(self, mapping_dict: Dict[int, List[int]]) -> str:
        """
        This method parses the mapping dictionary to a string in python format.

        Args:
            mapping_dict (Dict[List[int]]): The mapping dictionary.
        """
        res_str = "{"
        for key, value in mapping_dict.items():
            res_str += f"{key}: {value}, "
        res_str += "}"

        # validation
        assert eval(res_str) == mapping_dict, "Mapping string is not valid."

        # append the head
        res_str = self.mapping_head + res_str

        return res_str
