from abc import ABC

import pymongo
from bson import ObjectId

from sdl_orchestration import logger
from sdl_orchestration import sdl_config


def get_db():
    client = pymongo.MongoClient(sdl_config.atlas_url)
    db = client[sdl_config.database_name]
    logger.info("Connected to database")
    return db


class BaseDAO(ABC):
    device_id: ObjectId


database_client = get_db()
