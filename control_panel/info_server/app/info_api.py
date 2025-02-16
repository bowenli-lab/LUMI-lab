from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field
from pymongo import MongoClient
import os
import json
from bson import ObjectId

# import sys
# sys.path.append("..")

# from sdl_orchestration import sdl_config


class Sample(BaseModel):
    """
    A class to represent the sample data from the database.
    """

    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    last_updated: datetime
    wells: List[Dict]
    readings_list: List[Dict]

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: lambda v: str(v)}


class SDL_DB_API:
    def __init__(self):
        # Connect to MongoDB
        client = MongoClient(os.environ["ATLAS_URL"])
        database_name = "sdl-prod"
        db = client[database_name]
        self.sample_collection = db["sample"]
        self.reagent_collection = db["reagent"]

    def get_all_samples(self) -> List:
        """
        Get all the samples' ids and last updated time
        available in the database.

        Returns:
            List: A list of dictionaries containing
            the id and last updated time.
        """

        # filter and get samples that have a `readings_list` field
        entries = list(
            self.sample_collection.find(
                {"readings_list": {"$exists": True}}, {"last_updated": 1}
            )
        )
        result = [
            {"id": str(entry["_id"]), "last_updated": entry["last_updated"]}
            for entry in entries
        ]
        # list recent first
        result.sort(key=lambda x: x["last_updated"], reverse=True)
        return result

    def get_reading_result_by_id(self, sample_id: str):
        """
        Get the reading results for a specific sample.

        Args:
            sample_id (str): The id of the sample.

        Returns:
            Sample: A Pydantic model representing the sample
        """

        entry = self._get_reading_data_by_id(sample_id)
        data: Sample = Sample(**entry)
        return data

    def _get_reading_data_by_id(self, sample_id: str):
        """
        Get the reading data for a specific sample.

        Args:
            sample_id (str): The id of the sample.
        """
        sample_id = ObjectId(sample_id)
        entry = self.sample_collection.find_one({"_id": sample_id})
        return entry

    def get_mapped_result_by_id(self, sample_id: str):
        """
        Get the mapped result for a specific sample.

        Args:
            sample_id (str): The id of the sample.

        Returns:
            dict: A dictionary containing the mapped result.
        """

        entry = self._get_reading_data_by_id(sample_id)

        #  dict {'alphanum well' : {
        #      'reading': float,
        #      'components': {amines, isocyanide, carboxylic_acids, aldehydes}
        #  }}

        # alphanum is A1, A2, ... A12, B1, B2, ... H12
        alphanums = [f"{letter}{num}" for num in range(1, 13) for letter in "ABCDEFGH"]
        result = {
            key: {
                alphanum: {"reading": 0.0, "components": {}} for alphanum in alphanums
            }
            for key in range(len(entry["readings_list"]))
        }

        for entry_idx, reading in enumerate(entry["readings_list"]):
            for idx, well in enumerate(entry["wells"]):
                lipid_structure = well["lipid_structure"]
                # sort the lipid structure dict's key
                lipid_structure = {
                    k: lipid_structure[k] for k in sorted(lipid_structure)
                }

                # the idx is A1, B1, C1, ... H1, A2, B2, C2, ... H2, ...
                alphanum = alphanums[idx]
                result[entry_idx][alphanum]["components"] = lipid_structure
                result[entry_idx][alphanum]["reading"] = reading["results"][
                    alphanum[0]
                ][int(alphanum[1:]) - 1]

        return result

    def get_gain(self, sample_id: str):
        """
        Get the gain value for a specific sample.

        Args:
            sample_id (str): The id of the sample.

        Returns:
            float: The gain value.
        """
        entry = self._get_reading_data_by_id(sample_id)

        gain_values = {
            key: int(entry["readings_list"][key].get("gain_lum", -1))
            for key in range(len(entry["readings_list"]))
        }
        return gain_values

    def _get_reagent_entry(self):
        """
        Reagent field has only one entry in the database.
        """
        entry = self.reagent_collection.find_one()
        return entry

    def set_well_volume(self, well: str, value: float):
        """
        Set the reagent volume for a specific well and component.

        Args:
            well (str): The well location.
            value (float): The volume to set.
        """
        entry = self._get_reagent_entry()

        if not entry:
            raise ValueError("Reagent entry not found.")

        # alphabet to idx
        idx = (ord(well[0]) - 65) + (int(well[1]) - 1) * 8

        # update the volume
        try:
            entry["wells"][idx]["volume"] = value
        except Exception as e:
            print(e)
            raise ValueError(f"Well {well} not found.")

        # update the database
        self.reagent_collection.update_one(
            {"_id": entry["_id"]}, {"$set": {"wells": entry["wells"]}}
        )

    def set_component_volume(self, component: str, value: float):
        """
        Set the reagent volume for a specific component.

        Args:
            component (str): The component name.
            value (float): The volume to set.
        """
        entry = self._get_reagent_entry()

        if not entry:
            raise ValueError("Reagent entry not found.")

        is_updated = False
        for well in entry["wells"]:
            if well["name"] == component:
                well["volume"] = value
                is_updated = True

        if not is_updated:
            raise ValueError(f"Component {component} not found.")

        # update the database
        self.reagent_collection.update_one(
            {"_id": entry["_id"]}, {"$set": {"wells": entry["wells"]}}
        )

    def get_reagent(
        self,
    ):
        """
        Get the reagent information for a specific sample.

        Returns:
            List[Dict]: A list of dictionaries containing the well's location, volume, and lipid information.
        """
        entry = self._get_reagent_entry()

        if not entry:
            return []

        reagent_info = []

        for idx, well in enumerate(entry["wells"]):
            lipid_structure = well.get("lipid_structure", {})
            well_info = {
                # idx to alphabet
                "location": chr(65 + idx % 8) + str(idx // 8 + 1),
                "name": well.get("name"),
                "volume": well.get("volume"),
                "lipid_structure": {
                    "amines": lipid_structure.get("amines"),
                    "isocyanide": lipid_structure.get("isocyanide"),
                    "lipid_carboxylic_acid": lipid_structure.get(
                        "lipid_carboxylic_acid"
                    ),
                    "lipid_aldehyde": lipid_structure.get("lipid_aldehyde"),
                },
            }
            reagent_info.append(well_info)

        return reagent_info


def clear_reagent_volume():
    """
    Clear the reagent information in the database.
    """
    client = MongoClient(os.environ["ATLAS_URL"])
    database_name = "sdl-prod"
    db = client[database_name]
    reagent_collection = db["reagent"]
    reagent_collection.delete_many({})


class OpentronWatcher:
    """
    A class for watching the Opentron's camera.

    TODO: Pending implementation.
    """

    pass
