import datetime
from typing import Optional

from bson import ObjectId

from sdl_orchestration import logger
from sdl_orchestration.database.base_dao import BaseDAO, database_client
from sdl_orchestration.experiment.base_task import TaskStatus
from sdl_orchestration.experiment.samples.plate96well import Plate96Well


class TaskDAO(BaseDAO):

    def __init__(self, task_id: ObjectId):
        super().__init__()
        self.device_id = task_id
        self.client = database_client
        self.task_object_collection = self.client["task"]

    def create_entry(self, object_name: str,
                     experiment_id: Optional[ObjectId],
                     status: Optional[TaskStatus],
                     **kwargs):
        targets = kwargs.get("targets", None)
        if "targets" in kwargs and isinstance(targets, Plate96Well):
            kwargs["targets"] = targets.to_dict()
        entry = {
            "_id": self.device_id,
            "object_name": object_name,
            "status": str(status),
            "experiment_id": experiment_id,
            "created_at": datetime.datetime.now(),
            "last_updated": datetime.datetime.now(),
            **kwargs
        }
        try:
            self.task_object_collection.insert_one(entry)
            # logger.info(f"Task object recorded with id {self.device_id}")
        except Exception as e:
            logger.error(e)

    def update_status(self, status: TaskStatus) -> None:
        self.task_object_collection.update_one({"_id": self.device_id},
                                               {"$set": {
                                                   "status": str(status),
                                                   "last_updated": datetime.datetime.now()}})
