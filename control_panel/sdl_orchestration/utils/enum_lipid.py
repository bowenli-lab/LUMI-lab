from typing import Generator, List, Optional

from sdl_orchestration import logger
from sdl_orchestration.experiment import LipidStructure, sample_registry
from sdl_orchestration.experiment.samples.plate96well import Plate96Well


def lipid_list2plate96well_lst(lipid_list: List[LipidStructure],
                               num_control: int,
                               control_pos: Optional[List[str]] = None,
                               ) -> List[Plate96Well]:
    """
    This function converts a list of lipids to a list of Plate96Well.

    Args:
        lipid_list (List[LipidStructure]): a list of lipids.
        num_control (int): the number of control wells.
        control_pos (List[str], optional): the positions of control wells.
        if None, control wells will be placed at the beginning of each plate.

    Returns:
        List[Plate96Well]: a list of Plate96Well.
    """
    assert num_control >= 0, "num_control should be non-negative."
    if control_pos is None:
        # A1, A2, ... A12, B1, B2, ... H12 for 96-well plate
        # we get the first num_control wells
        total_str = "ABCDEFGH"
        control_pos = [f"{total_str[i // 12]}{i % 12 + 1}"
                       for i in range(num_control)]
    else:
        assert num_control == len(control_pos), \
            "num_control should be equal to the length of control_pos."

    plate96well_list = [Plate96Well(is_reagent=False) for _ in
                        range((len(lipid_list) + num_control) // 96 + 1)]

    for plate in plate96well_list:
        control_list = []
        for _ in control_pos:
            # append control wells
            control_list.append(LipidStructure(is_control=True))
        plate.add_targets(control_list,
                          target_well_strs=control_pos,
                          sync=False)

    current_plate96well = plate96well_list.pop(0)
    res_plate96well_list = [current_plate96well]
    for lipid in lipid_list:
        if len(current_plate96well) >= current_plate96well.max_capacity:
            # if the current plate is full, move to the next plate
            # save the current plate to list
            current_plate96well = plate96well_list.pop(0)
            res_plate96well_list.append(current_plate96well)
        current_plate96well.add_target(lipid, sync=False)

    # sync all
    for plate in res_plate96well_list:
        plate.sync()

    return res_plate96well_list


class LipidStructureEnumerator:
    """
    This class is responsible for enumerating all possible lipid structures.
    """

    def __init__(self):
        reagent_plate = sample_registry.reagent
        self.amines = reagent_plate.get_lipid_by_type("A")
        self.isocyanide = reagent_plate.get_lipid_by_type("B")
        self.lipid_carboxylic_acid = reagent_plate.get_lipid_by_type("C")
        self.lipid_aldehyde = reagent_plate.get_lipid_by_type("D")

        logger.debug(f"Enumerating lipid structures: "
                     f"amines: ({len(self.amines)}),"
                     f"isocyanide: ({len(self.isocyanide)}),"
                     f"lipid_carboxylic_acid: ({len(self.lipid_carboxylic_acid)}), "
                     f"lipid_aldehyde: ({len(self.lipid_aldehyde)})")

    def _yield_lipid_structure(self):
        """
        This method yields all the possible lipid structures.

        Yields:
            LipidStructure: a lipid structure.
        """
        for amine in self.amines:
            for isocyanide in self.isocyanide:
                for lipid_carboxylic_acid in self.lipid_carboxylic_acid:
                    for lipid_aldehyde in self.lipid_aldehyde:
                        yield LipidStructure(
                            amines=str(amine),
                            isocyanide=str(isocyanide),
                            lipid_carboxylic_acid=str(lipid_carboxylic_acid),
                            lipid_aldehyde=str(lipid_aldehyde)
                        )

    def get_generator(self) -> Generator[LipidStructure,
    None, None]:
        """
        This method returns a generator for all the possible lipid structures.
        """
        return self._yield_lipid_structure()
