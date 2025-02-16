from typing import Any, Dict, List, Optional, Tuple

from bson.objectid import ObjectId

from sdl_orchestration import logger, sdl_config
from sdl_orchestration.database.daos.sample_dao import SampleDAO
from sdl_orchestration.experiment import BaseSample, LipidStructure, \
    SampleStatus, Well

PLATE_96_ROW_NUM = 8
PLATE_96_COL_NUM = 12
PLATE_96_NUM_WELL = PLATE_96_ROW_NUM * PLATE_96_COL_NUM


class Plate96Well(BaseSample):
    """
    This class is a sample of a 96 well plate.

    Args:
        sample_id (Optional[ObjectId]): the sample id.
        sample_status (SampleStatus): the sample status.
        sample_location (Optional[str]): the sample location.
        sample_description (Optional[str]): the sample description.
        experiment_id (Optional[ObjectId]): the experiment id.
        is_reagent (bool): whether the sample is a reagent.

    Attributes:
        is_reagent (bool): whether the sample is a reagent.
        sample_location (Optional[str]): the sample location.
        sample_description (Optional[str]): the sample description.
        experiment_id (Optional[ObjectId]): the experiment id.
        wells (List[Well]): the list of wells in the 96 well plate.
        max_capacity (int): the maximum capacity of the 96 well plate.
        dao (SampleDAO): the sample DAO.
    """

    def __init__(
            self,
            sample_id: Optional[ObjectId] = None,
            sample_status: SampleStatus = SampleStatus.READY,
            sample_location: Optional[str] = None,
            sample_description: Optional[str] = None,
            experiment_id: Optional[ObjectId] = None,
            is_reagent: bool = False,
    ):
        if sample_id is None:
            # avoid replication of sample_id
            sample_id = ObjectId()
        super().__init__(sample_id, sample_status)

        self.is_reagent = is_reagent
        self.sample_location = sample_location
        self.sample_description = sample_description
        self.experiment_id = experiment_id

        self.reading_info_list = []
        self.results_list = []

        self.wells = [Well(coordinate=num) for num in range(PLATE_96_NUM_WELL)]
        self.max_capacity = PLATE_96_NUM_WELL

        if is_reagent:
            logger.info(f"Configuring reagents in the 96 well plate")
            self._config_reagent()

        logger.info("96 well plate created.")

        self.dao = SampleDAO(self.sample_id, is_reagent, experiment_id)
        query_res = self.dao.create_entry(
            sample_status, sample_location, sample_description, self.wells
        )
        if is_reagent and query_res:
            # load the reagent from the database
            self._load_reagent(query_res)

    def _load_reagent(self, query_res: Dict[str, Any]) -> None:
        """
        Load the reagent from the database.
        """
        wells = query_res["wells"]
        for well, well_dict in zip(self.wells, wells):
            well.name = well_dict["name"]
            well.volume = well_dict["volume"]
            well.concentration = well_dict["concentration"]
            well.lipid_structure = LipidStructure().from_dict(
                well_dict["lipid_structure"]
            )
            well.occupied = well_dict["occupied"]
        self.sample_id = query_res["_id"]
        logger.info(
            f"Reagent loaded from the database with id {self.sample_id}")
        logger.info(f"Reagent: {self}")

    def get_all_lipids(self) -> List[LipidStructure]:
        """
        Get all the lipids in the 96 well plate.
        """
        return [well.lipid_structure for well in self.wells if well.occupied]

    def get_lipid_by_type(self, lipid_type: str) -> List[LipidStructure]:
        """
        Get the list of lipids by the type.

        Args:
            lipid_type (str): the type of the lipid.
        """
        return [
            well.lipid_structure
            for well in self.wells
            if (
                    not well.is_control
                    and well.occupied
                    and well.lipid_structure.get_type() == lipid_type
            )
        ]

    @staticmethod
    def _get_well_index(row: int, col: int) -> int:
        """
        Get the index of the well in the 96 well plate.

        Args:
            row (int): the row of the well.
            col (int): the column of the well.
        """
        return col * PLATE_96_ROW_NUM + row

    def index2alphabet(self, index: int) -> str:
        """
        Convert the index to the alphabet format.
        """
        col = index // PLATE_96_ROW_NUM
        row = index % PLATE_96_ROW_NUM
        return self.coordinate2alphabet(row, col)

    def get_well(self, row: int, col: int) -> Well:
        """
        Get the well at the given row and column.
        """
        return self.wells[self._get_well_index(row, col)]

    def set_well(self, row: int, col: int, well: Well) -> None:
        """
        Set the well at the given row and column.
        """
        self.wells[self._get_well_index(row, col)] = well

    def add_target(
            self,
            lipid: LipidStructure,
            volume: Optional[float] = None,
            concentration: float = None,
            target_well_str: Optional[str] = None,
            sync: bool = True,
    ) -> None:
        """
        Add a target to the next available well in 96 well plate if
        target_well_str is None. Otherwise, add the target to the specified well.

        This method also syncs the 96 well plate with the database.

        Args:
            lipid (LipidStructure): the lipid to add.
            volume (Optional[float]): the volume of the lipid.
            concentration (float): the concentration of the
                lipid.
            target_well_str (Optional[str]): the target well location in the alphabet+number format.
            sync (bool): whether to sync the 96 well plate with the database.
        """
        self._add_target(lipid, volume, concentration, target_well_str)
        if sync:
            self._sync_dao()

    def _sample_target(
            self, lipid: LipidStructure, volume: Optional[float] = None
    ) -> None:
        """
        Sample a target from the 96 well plate.

        This method finds the lipid first and then samples the lipid.
        """
        raise NotImplementedError

    def _add_target(
            self,
            lipid: LipidStructure,
            volume: Optional[float] = None,
            concentration: float = None,
            target_well_str: Optional[str] = None,
    ) -> None:
        """
        This method adds a target to the next available well in 96 well plate.
        """
        # TODO: simplify this method
        if target_well_str is not None:
            row, col = self.alphabet2coordinate(target_well_str)
            well = self.get_well(row, col)
            if well.occupied:
                raise Exception(f"Well {target_well_str} is occupied.")
            well.name = str(lipid)
            well.lipid_structure = lipid
            well.volume = volume
            well.occupied = True
            if lipid.is_control:
                well.is_control = True
            if concentration:
                well.concentration = concentration
            return None
        for well in self.wells:
            if not well.occupied:
                well.name = str(lipid)
                well.lipid_structure = lipid
                well.volume = volume
                well.occupied = True
                if lipid.is_control:
                    well.is_control = True
                if concentration:
                    well.concentration = concentration
                return None
        raise Exception("No available well in the 96 well plate.")

    def add_targets(
            self,
            lipids: List[LipidStructure],
            volumes: Optional[List[Optional[float]]] = None,
            concentrations: Optional[List[Optional[float]]] = None,
            target_well_strs: Optional[List[Optional[str]]] = None,
            sync: bool = True,
    ) -> None:
        """
        Add targets to the next available wells in 96 well plate.

        This method also syncs the 96 well plate with the database.

        Args:
            lipids (List[LipidStructure]): the list of lipids to add.
            volumes (List[float]): the list of volumes of the lipids.
            concentrations (Optional[List[float]]): the list of concentrations of the lipids.
            target_well_strs (Optional[List[str]]): the list of target well locations in the alphabet+number format.
            sync (bool): whether to sync the 96 well plate with the database.
        """
        if volumes is None:
            volumes = [None] * len(lipids)
        if concentrations is None:
            concentrations = [None] * len(lipids)

        for lipid, volume, concentration, target_well_str in zip(lipids,
                                                                 volumes,
                                                                 concentrations,
                                                                 target_well_strs):
            self._add_target(lipid, volume, concentration, target_well_str)

        if sync:
            self._sync_dao()

    def sample_targets(
            self, lipids: List[LipidStructure], volumes: List[Optional[float]]
    ) -> None:
        """
        Sample a target from the 96 well plate.
        """
        pass

    def _validate_safety(self, ) -> None:
        """
        Validate the safety of the 96 well plate by rule checking the volume.

        RULE 1: the volume of the well should be larger than the safety volume.
        RULE 2: the volume of the well should be smaller than the cap volume.
        """
        for well in self.wells:
            if well.volume < sdl_config.liquid_sampler_safe_volume:
                raise Exception(f"Volume of well {well.coordinate} is too low.")
            if well.volume > sdl_config.liquid_sampler_safe_cap:
                raise Exception(
                    f"Volume of well {well.coordinate} is too high.")

    def fill_target_by_alphanum_well(self, well: str,
                                     volume: Optional[float] = None) -> None:
        """
        Fill the target in the well in the 96 well plate.

        Args:
            well (str): the well location in the alphabet+number format.
            volume (Optional[float]): the volume to fill, in unit of uL.
        """
        row, col = self.alphabet2coordinate(well)
        well = self.get_well(row, col)
        well.volume += volume

    def fill_targets_by_alphanum_well(
            self, wells: List[str],
            volumes: List[Optional[float]]
    ) -> None:
        """
        Fill a multiple targets from the 96 well plate. Update the volume of
        each well.
        """
        for well, volume in zip(wells, volumes):
            self.fill_target_by_alphanum_well(well, volume)
        self._validate_safety()
        self._sync_dao()

    def sample_targets_by_alphanum_well(
            self,
            wells: List[str],
            volumes: List[Optional[float]]
    ) -> None:
        """
        Extract volumes of multiple targets from the 96 well plate. Update the volume of
        each well.

        Args:
            wells (List[str]): the list of well locations in the alphabet+number format.
            volumes (List[Optional[float]]): the list of volumes to subtract.
        """

        for well, volume in zip(wells, volumes):
            self.sample_target_by_alphanum_well(well, volume)
        self._validate_safety()
        self._sync_dao()

    def sample_target_by_alphanum_well(
            self, well: str, volume: Optional[float]
    ) -> None:
        """
        Use sample in the well in the 96 well plate.
        Update the volume of the well.

        Args:
            well (str): the well location in the alphabet+number format.
            volume (Optional[float]): the volume to subtract, in unit of uL.
        """
        row, col = self.alphabet2coordinate(well)
        well = self.get_well(row, col)
        well.volume -= volume

    def _sync_dao(self) -> None:
        """
        Sync the 96 well plate with the database.
        """
        self.dao.update_entry(
            status=self.sample_status,
            location=str(self.sample_location),
            wells=self.wells,
            readings_list=self.reading_info_list,
            results_list=self.results_list,
        )

    def sync(self) -> None:
        """
        Sync the 96 well plate with the database.
        """
        self._sync_dao()

    def get_available_num_wells(self) -> int:
        """
        Get the number of available wells in the 96 well plate.
        """
        return PLATE_96_NUM_WELL - len(self)

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """
        Convert the 96 well plate to a dictionary
        where the key is the well index and the value is the well dictionary.
        """
        res = {}
        for well in self.wells:
            res[str(well.coordinate)] = well.to_dict()
        return res

    def __len__(self) -> int:
        """
        Get the number of occupied wells in the 96 well plate.
        """
        counter = 0
        for well in self.wells:
            if well.occupied:
                counter += 1
        return counter

    def locate_reagent(self, query: LipidStructure) -> Optional[
        Tuple[int, int]]:
        """
        locate the coordinate of the queried reagent in the 96 well plate.

        Args:
            query (LipidStructure): the reagent to locate.

        Returns:
            Optional[Tuple[int, int]]: the coordinate of the reagent if found.
                None otherwise.
        """

        for row in range(PLATE_96_ROW_NUM):
            for col in range(PLATE_96_COL_NUM):
                well = self.get_well(row, col)
                if well.lipid_structure == query:
                    return row, col
        return None

    def locate_reagent_well_coord(self, query: LipidStructure) -> Optional[int]:
        """
        locate the well of the queried reagent in the 96 well plate.

        Args:
            query (LipidStructure): the reagent to locate.

        Returns:
            Optional[Well]: the well of the reagent if found.
                None otherwise.
        """

        for row in range(PLATE_96_ROW_NUM):
            for col in range(PLATE_96_COL_NUM):
                well = self.get_well(row, col)
                if well.lipid_structure == query:
                    return self._get_well_index(row, col)
        return None

    def get_lipid_list(self) -> List[LipidStructure]:
        """
        Get the list of lipids in the 96 well plate.
        """
        return [well.lipid_structure for well in self.wells if well.occupied]

    def get_lipid_dict(self) -> Dict[int, LipidStructure]:
        """
        Get the dictionary of lipids in the 96 well plate.

        Returns:
            Dict[int, LipidStructure]: the dictionary of lipids in
             the 96 well plate.
        """
        res = {}
        for well in self.wells:
            if well.occupied:
                res[well.coordinate] = well.lipid_structure
        return res

    @staticmethod
    def coordinate2alphabet(row: int, col: int) -> str:
        """
        Convert the coordinate to the alphabet format.
        """
        return f"{chr(ord('A') + row)}{col + 1}"

    @staticmethod
    def alphabet2coordinate(alphabet: str) -> Tuple[int, int]:
        """
        Convert the alphabet format to the coordinate.
        """
        return ord(alphabet[0]) - ord("A"), int(alphabet[1:]) - 1

    def _config_reagent(self, coord_format="alpha_num") -> None:
        """
        Configure the reagent using the config file as specified in the
        environment variable.

        Assume the reagent configuration is stored in the REAGENT_CONFIG
        environment variable.
        """
        reagents = sdl_config.reagent_config
        logger.info(f"Reagent configuration: {sdl_config.reagent_config}")

        # skip the first row as it is the motor for plate
        iter_reagents = reagents.iterrows()

        for index, row in iter_reagents:
            location = row["Well_cor"]
            reagent = row["Reagent"]
            volume = row["Volume"]

            # skip if nan
            if reagent != reagent:
                continue

            # Assume the first letter is the type
            reagent_type = row["Reagent"][0]

            logger.debug(f"Configuring reagent at {location}, {reagent}")
            lipid = LipidStructure(
                amines=reagent if reagent_type == "A" else None,
                isocyanide=reagent if reagent_type == "B" else None,
                lipid_aldehyde=reagent if reagent_type == "C" else None,
                lipid_carboxylic_acid=reagent if reagent_type == "D" else None,
            )

            if coord_format == "alpha_num":
                row, col = self.alphabet2coordinate(location)
            else:
                raise NotImplementedError

            well = self.get_well(row, col)
            well.name = str(lipid)
            well.lipid_structure = lipid
            well.volume = float(volume)
            well.occupied = True

    def get_well_by_alphanum(self, location: str) -> Well:
        """
        Get the well by the location in the alphabet+number format.
        """
        row, col = self.alphabet2coordinate(location)
        return self.get_well(row, col)

    def map_target2wells(
            self,
            target: LipidStructure,
    ) -> List[int]:
        """
        Map the target to the wells in the 96 well plate.

        Args:
            target (LipidStructure): the targets to map. Assume each
                target has four components (amines, isocyanide, lipid_carboxylic_acid, lipid_aldehyde).

        Returns:
            List[int]: the well's int coord mapped to the targets.
        """

        wells = []

        if target.is_control:
            return wells

        # We map the head group, tail A, tail B, and lipid_aldehyde to the wells
        # respectively.

        # head group
        amines = target.amines
        wells.append(
            self.locate_reagent_well_coord(LipidStructure(amines=amines)))

        # tail A
        isocyanide = target.isocyanide
        wells.append(
            self.locate_reagent_well_coord(
                LipidStructure(isocyanide=isocyanide))
        )

        # tail B
        lipid_carboxylic_acid = target.lipid_carboxylic_acid
        wells.append(
            self.locate_reagent_well_coord(
                LipidStructure(lipid_carboxylic_acid=lipid_carboxylic_acid)
            )
        )

        # lipid_aldehyde
        lipid_aldehyde = target.lipid_aldehyde
        wells.append(
            self.locate_reagent_well_coord(
                LipidStructure(lipid_aldehyde=lipid_aldehyde)
            )
        )

        # check if all the wells are found
        if None in wells:
            logger.error(
                f"Not all the wells are found for" f" the target: {target}")
            raise Exception(
                f"Not all the wells are found for the target: {target}")

        # drop all None's in the list
        wells = [well for well in wells if well is not None]

        return wells

    def copy(self) -> "Plate96Well":
        """
        Copy the 96 well plate.
        """
        plate = Plate96Well(
            sample_id=self.sample_id,
            sample_status=self.sample_status,
            sample_location=self.sample_location,
            sample_description=self.sample_description,
            experiment_id=self.experiment_id,
            is_reagent=self.is_reagent,
        )

        for well, new_well in zip(self.wells, plate.wells):
            new_well.name = well.name
            new_well.volume = well.volume
            new_well.concentration = well.concentration
            new_well.lipid_structure = well.lipid_structure.copy()
            new_well.occupied = well.occupied

        return plate

    def add_reading_info(
            self, reading_info: Dict[str, str], results: Dict[str, str]
    ) -> None:
        """
        Add reading information to the 96 well plate.

        Args:
            reading_info (Dict[str, str]): the reading information.
            results (Dict[str, str]): the results.
        """
        self.reading_info_list.append(reading_info)
        self.results_list.append(results)
        logger.info(f"Reading info added to the 96 well plate {self.sample_id}")
        logger.info(f"Reading info: {reading_info}")
        logger.info(f"Results: {results}")
        self._sync_dao()

    def __str__(self):
        """
        Brief description of the 96 well plate.
        """
        str_string = f"96 well plate ({len(self)} occupied)"
        if len(self.results_list) > 0:
            str_string += f" with {len(self.results_list)} readings"
        return str_string

    def __repr__(self):
        """
        Representation of the 96 well plate.
        """
        return self.__str__()
