import datetime
from typing import Any, List, Optional

from bson import ObjectId

from sdl_orchestration import logger
from sdl_orchestration.database.base_dao import BaseDAO, database_client
from sdl_orchestration.experiment.base_sample import LipidStructure
from sdl_orchestration.experiment.samples.plate96well import Plate96Well


class ExperimentDAO(BaseDAO):
    device_id: ObjectId

    def __init__(self, experiment_id: ObjectId):
        super().__init__()
        self.device_id = experiment_id
        self.client = database_client
        self.experiment_object_collection = self.client["experiment"]

    def create_entry(self, object_name: str,
                     status: Optional[Any],
                     targets: Optional[Plate96Well] = None,
                     experiment_index: Optional[int] = None,
                     **kwargs):
        if targets is None:
            targets = []
            logger.info(f"experiment {self.device_id}, No targets provided.")

        targets = targets.to_dict()
        entry = {
            "_id": self.device_id,
            "object_name": object_name,
            "status": str(status),
            "targets": targets,
            "current_task_id": "",
            "created_at": datetime.datetime.now(),
            "last_updated": datetime.datetime.now(),
            "experiment_index": experiment_index,
            **kwargs
        }
        try:
            self.experiment_object_collection.insert_one(entry)
            logger.info(f"Experiment object recorded with id {self.device_id}")
        except Exception as e:
            logger.ctritical(e, exc_info=True)

    def update_status(self, status: Any) -> None:
        self.experiment_object_collection.update_one({"_id": self.device_id},
                                                     {"$set": {
                                                         "status": str(status),
                                                         "last_updated": datetime.datetime.now()}})

    def update_current_task(self, task_id: ObjectId) -> None:
        self.experiment_object_collection.update_one({"_id": self.device_id},
                                                     {"$set": {
                                                         "current_task_id": str(
                                                             task_id),
                                                         "last_updated": datetime.datetime.now()}})

    def update_state(self, state: dict) -> None:
        self.experiment_object_collection.update_one({"_id": self.device_id},
                                                     {"$set": state})

    def get_state(self) -> dict:
        return self.experiment_object_collection.find_one({"_id": self.device_id})
