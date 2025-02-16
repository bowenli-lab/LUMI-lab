import datetime
from typing import Any, List, Optional

from bson import ObjectId

from sdl_orchestration import logger
from sdl_orchestration.database.base_dao import BaseDAO, database_client
from sdl_orchestration.experiment.base_sample import Well


class SampleDAO(BaseDAO):
    device_id: ObjectId

    def __init__(self, sample_id: ObjectId, is_reagent: bool,
                 experiment_id: Optional[ObjectId] = None):
        super().__init__()
        self.device_id = sample_id
        self.client = database_client

        self.experiment_id = experiment_id
        if is_reagent:
            self.sample_object_collection = self.client["reagent"]
        else:
            self.sample_object_collection = self.client["sample"]
        self.is_reagent = is_reagent

    def create_entry(self,
                     status: Optional[Any],
                     location: Optional[str],
                     description: Optional[str],
                     wells: List[Well],
                     **kwargs):
        if self.is_reagent:
            query_res = self.sample_object_collection.find_one({"object_name":
                                                                "reagent"})
            if query_res:
                self.device_id = query_res["_id"]
                logger.info(
                    f"Reagent already exists with id {query_res['_id']},"
                    f"inheriting from the existing object.")
                return query_res

        wells = [well.to_dict() for well in wells]
        entry = {
            "_id": self.device_id,
            "status": str(status),
            "object_name": "reagent" if self.is_reagent else "sample",
            "location": location,
            "description": description,
            "wells": wells,
            "created_at": datetime.datetime.now(),
            "last_updated": datetime.datetime.now(),
            **kwargs
        }
        if self.experiment_id:
            entry["experiment_id"] = self.experiment_id
        try:
            self.sample_object_collection.insert_one(entry)
            logger.info(f"Sample object recorded with id {self.device_id}")
        except Exception as e:
            logger.error(e)
        return None

    def update_entry(self,
                     object_name: Optional[str] = None,
                     status: Optional[Any] = None,
                     location: Optional[str] = None,
                     description: Optional[str] = None,
                     wells: Optional[List[Well]] = None,
                     readings_list: Optional[List[dict]] = None,
                     results_list: Optional[List[dict]] = None,
                     **kwargs):
        update = {
            "last_updated": datetime.datetime.now()
        }
        if object_name:
            update["object_name"] = object_name
        if status:
            update["status"] = str(status)
        if location:
            update["location"] = location
        if description:
            update["description"] = description
        if wells:
            update["wells"] = [well.to_dict() for well in wells]
        if readings_list:
            update["readings_list"] = readings_list
        if results_list:
            update["results_list"] = results_list
        update.update(kwargs)
        try:
            self.sample_object_collection.update_one({"_id": self.device_id},
                                                     {"$set": update})
            logger.info(f"Sample object updated with id {self.device_id}")
        except Exception as e:
            logger.error(e)
