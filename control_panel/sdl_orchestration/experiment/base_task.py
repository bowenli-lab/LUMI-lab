from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from bson.objectid import ObjectId

from .base_sample import BaseSample
from .. import logger
from ..notification.client import notification_client


class TaskStatus(Enum):
    """The status of the task."""

    WAITING = "WAITING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    BLOCKED = "BLOCKED"


class BaseTask(ABC):
    """
    This is the base class for all tasks. It is an abstract class and
     should not be instantiated directly.
    """

    def __init__(
        self,
        task_id: Optional[ObjectId] = None,
        experiment_id: Optional[ObjectId] = None,
        precursor_task_id: List[Optional[ObjectId]] | None = None,
        task_status: TaskStatus = TaskStatus.WAITING,
        priority: int = 0,
        samples: List[Optional[BaseSample]] | None = None,
        targets: Optional[BaseSample] | None = None,
        *args,
        **kwargs,
    ):
        if task_id is None:
            task_id = ObjectId()

        if precursor_task_id is None:
            precursor_task_id = []

        if samples is None:
            samples = []

        self.task_id = task_id
        self.precursor_task_id = precursor_task_id
        self.task_status = task_status
        self.priority = priority
        self.samples = samples
        self.experiment_id = experiment_id
        self.targets = targets

        self.task_dao = None

    @abstractmethod
    def run(self):
        """
        This method is called to run the task.
        """
        raise NotImplementedError

    @abstractmethod
    def check_condition(self, *args, **kwargs):
        """
        This method is called to check the condition of the task.
        """
        raise NotImplementedError

    def _update_status(self, status: TaskStatus):
        """
        This method is called to update the status of the task.
        """
        raise NotImplementedError

    def fail(self, info: Optional[str] = None):
        """
        This method is called when the task fails.
        """
        logger.error(f"Task {self.task_id} failed.")
        self._update_status(TaskStatus.ERROR)

        notification_client.send_notification(
            message_type="error",
            mention_all=True,
            message=f"Task {self.task_id} failed.",
            error_message=str(info),
        )

    def auto_fail(self):
        """
        This method is called when the task fails.
        """
        logger.error(f"Task {self.task_id} failed " f"(by experiment manager).")
        self._update_status(TaskStatus.ERROR)
