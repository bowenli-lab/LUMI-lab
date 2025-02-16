from enum import Enum
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from bson.objectid import ObjectId


class SampleStatus(Enum):
    """
    The status of the sample
    """

    EMPTY = "EMPTY"
    READY = "READY"
    PLANNED = "PLANNED"
    WAITING = "WAITING"
    OCCUPIED = "OCCUPIED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class BaseSample:
    """
    This is the base class for the sample, all the samples should
     inherit from this class.
    """

    sample_id: Optional[ObjectId] = None
    sample_status: SampleStatus = SampleStatus.WAITING

    def __init__(
            self,
            sample_id: Optional[ObjectId] = None,
            sample_status: SampleStatus = SampleStatus.WAITING,
            *args,
            **kwargs,
    ):
        self.sample_id = sample_id
        self.sample_status = sample_status

    def get_identifier(self) -> str:
        """
        Get the identifier of the sample.
        """
        return str(self.sample_id)


@dataclass
class LipidStructure:
    amines: Optional[str] = None
    isocyanide: Optional[str] = None
    lipid_carboxylic_acid: Optional[str] = None
    lipid_aldehyde: Optional[str] = None
    is_control: bool = False

    def __str__(self):
        #     iff one field is filled, return the field
        all_fields = [
            self.amines,
            self.isocyanide,
            self.lipid_carboxylic_acid,
            self.lipid_aldehyde,
        ]
        if all_fields.count(None) == 4:
            if self.is_control:
                return "Control"
            return "Empty"
        elif all_fields.count(None) == 0:
            return (
                f"({self.amines},{self.isocyanide},"
                f"{self.lipid_carboxylic_acid},{self.lipid_aldehyde})"
            )
        elif self.amines:
            return self.amines
        elif self.isocyanide:
            return self.isocyanide
        elif self.lipid_carboxylic_acid:
            return self.lipid_carboxylic_acid
        elif self.lipid_aldehyde:
            return self.lipid_aldehyde

        #     if more than one field is filled, return all the fields

    def __repr__(self):
        if (
                self.amines
                and self.isocyanide
                and self.lipid_carboxylic_acid
                and self.lipid_aldehyde
        ):
            return (
                f"({self.amines},{self.isocyanide},"
                f"{self.lipid_carboxylic_acid},{self.lipid_aldehyde})"
            )
        elif (
                self.amines
                or self.isocyanide
                or self.lipid_carboxylic_acid
                or self.lipid_aldehyde
        ):
            non_empty_field = [
                field
                for field in [
                    self.amines,
                    self.isocyanide,
                    self.lipid_carboxylic_acid,
                    self.lipid_aldehyde,
                ]
                if field
            ]
            return f"Reagent ({str(non_empty_field)})"
        elif self.is_control:
            return "Control"
        else:
            return "Empty"

    def __eq__(self, other):
        if not isinstance(other, LipidStructure):
            return False
        return (
                self.amines == other.amines
                and self.isocyanide == other.isocyanide
                and self.lipid_carboxylic_acid == other.lipid_carboxylic_acid
                and self.lipid_aldehyde == other.lipid_aldehyde
        )

    def copy(self) -> "LipidStructure":
        return LipidStructure(
            amines=self.amines,
            isocyanide=self.isocyanide,
            lipid_carboxylic_acid=self.lipid_carboxylic_acid,
            lipid_aldehyde=self.lipid_aldehyde,
        )

    def is_empty(self) -> bool:
        """
        Check if the lipid structure is empty.
        """
        return not (
                self.amines
                or self.isocyanide
                or self.lipid_carboxylic_acid
                or self.lipid_aldehyde
        )

    def get_lipid_structures(self) -> tuple[
        "LipidStructure",
        "LipidStructure",
        "LipidStructure",
        "LipidStructure"
    ]:
        """
        Get the lipid structure for each field.

        Returns:
            Tuple[LipidStructure]: a tuple of lipid structures.
        """
        amines = LipidStructure(amines=self.amines)
        isocyanide = LipidStructure(isocyanide=self.isocyanide)
        lipid_carboxylic_acid = LipidStructure(
            lipid_carboxylic_acid=self.lipid_carboxylic_acid
        )
        lipid_aldehyde = LipidStructure(lipid_aldehyde=self.lipid_aldehyde)

        return amines, isocyanide, lipid_carboxylic_acid, lipid_aldehyde

    def to_dict(self):
        return {
            "amines": self.amines,
            "isocyanide": self.isocyanide,
            "lipid_carboxylic_acid": self.lipid_carboxylic_acid,
            "lipid_aldehyde": self.lipid_aldehyde,
        }

    def from_dict(self,
                  lipid_structure_dict: Dict[str, Any]) -> "LipidStructure":
        if not lipid_structure_dict:
            return LipidStructure()
        return LipidStructure(
            amines=lipid_structure_dict["amines"],
            isocyanide=lipid_structure_dict["isocyanide"],
            lipid_carboxylic_acid=lipid_structure_dict["lipid_carboxylic_acid"],
            lipid_aldehyde=lipid_structure_dict["lipid_aldehyde"],
        )

    def get_type(self) -> str:
        """
        Get the type of the lipid structure.
        """
        # check if only one field is filled
        if (
                sum(
                    [
                        bool(self.amines),
                        bool(self.isocyanide),
                        bool(self.lipid_carboxylic_acid),
                        bool(self.lipid_aldehyde),
                    ]
                )
                == 1
        ):
            if self.amines:
                return "A"
            elif self.isocyanide:
                return "B"
            elif self.lipid_carboxylic_acid:
                return "C"
            elif self.lipid_aldehyde:
                return "D"
        return ""


@dataclass
class Reading:
    """
    This class defines a reading for each well.
    """

    def __init__(
            self,
            one_glo: Optional[float] = None,
    ):
        """
        Args:
            one_glo (float): the reading of the well
        """
        self.one_glo = one_glo

    def to_dict(self) -> Dict[str, Any]:
        """
        Transform the reading to a dictionary.
        """
        return {"one_glo": self.one_glo}


class Well:
    """
    This class defines a well.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            volume: Optional[float] = None,
            concentration: Optional[float] = None,
            lipid_structure: Optional[LipidStructure] = None,
            occupied: Optional[bool] = False,
            coordinate: Optional[int] = None,
            is_control: Optional[bool] = False,
            reading: Optional[Reading] = None,
    ):
        """
        name (str): the chemical in the well
        volume (float): the volume of the chemical in the well
        concentration (float): the concentration of the chemical in the well
        lipid_structure (LipidStructure): the lipid structure of the chemical
                                    in the well
        occupied (bool): whether the well is occupied
        """
        self.name = name
        self.volume = volume
        self.concentration = concentration
        self.lipid_structure = lipid_structure
        self.occupied = occupied
        self.coordinate = coordinate
        self.is_control = is_control
        self.reading = reading

    def add_reading(self, one_glo: float) -> None:
        """
        Add a reading to the well.

        Args:
            one_glo (float): the reading of the well
        """
        self.reading = Reading(one_glo=one_glo)

    def __str__(self) -> str:
        multi_row_str = f"Name: {self.name}\n" if self.name else ""
        multi_row_str += f"Volume: {self.volume}\n" if self.volume else ""
        multi_row_str += (
            f"Concentration: {self.concentration}\n" if self.concentration else ""
        )
        multi_row_str += (
            f"Lipid Structure: {self.lipid_structure}\n" if self.lipid_structure else ""
        )
        multi_row_str += f"Occupied: {self.occupied}\n" if self.occupied else ""
        return multi_row_str

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Transform the well to a dictionary.
        """

        well_dict = {
            "name": str(self.name),
            "volume": self.volume,
            "concentration": self.concentration,
            "lipid_structure": (
                self.lipid_structure.to_dict() if self.lipid_structure else None
            ),
            "occupied": self.occupied,
        }

        if self.reading:
            well_dict["reading"] = self.reading.to_dict()

        return well_dict
