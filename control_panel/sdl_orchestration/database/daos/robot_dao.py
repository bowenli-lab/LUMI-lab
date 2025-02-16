import datetime
from typing import Optional

from bson import ObjectId

from sdl_orchestration import logger
from sdl_orchestration.database.base_dao import BaseDAO, database_client


class RobotDAO(BaseDAO):
    device_id: ObjectId

    def __init__(self, device_id: ObjectId):
        super().__init__()
        self.object_name = None
        self.device_id = device_id
        self.client = database_client
        self.robot_object_collection = self.client["devices"]
        self.robot_execution_collection = self.client["devices_execution"]

    def create_entry(self, object_name: str, status, **kwargs) -> ObjectId:
        self.object_name = object_name

        # if exist the same object_name then update the status
        query_res = self.robot_object_collection.find_one({"object_name": object_name})
        if query_res:
            logger.info(f"Object already exists with id {query_res['_id']}")
            self.device_id = query_res["_id"]
            self.update_status(status)
            return self.device_id

        entry = {
            "_id": self.device_id,
            "object_name": object_name,
            "status": str(status),
            "created_at": datetime.datetime.now(),
            "last_updated": datetime.datetime.now(),
            **kwargs,
        }
        try:
            self.robot_object_collection.insert_one(entry)
            logger.info(f"Feeder object recorded with id {self.device_id}")
        except Exception as e:
            logger.error(e)

        return self.device_id

    def log_step(self, step: str, task_id: Optional[ObjectId],
                 experiment_id: Optional[ObjectId], ):
        entry = {
            "_id": ObjectId(),
            "device_id": self.device_id,
            "object_name": self.object_name,
            "step": step,
            "task_id": task_id,
            "experiment_id": experiment_id,
            "created_at": datetime.datetime.now(),
            "last_updated": datetime.datetime.now(),

        }
        self.robot_execution_collection.insert_one(entry)

    def update_status(self, status):
        (self.robot_object_collection.
         update_one({"_id": self.device_id},
                    {"$set": {"status": str(status),
                              "last_updated": datetime.datetime.now()}}))
