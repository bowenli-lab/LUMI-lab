import time
from typing import Any, List, Optional

from bson.objectid import ObjectId

from sdl_orchestration import logger
from sdl_orchestration.database.daos import TaskDAO
from sdl_orchestration.experiment.base_sample import BaseSample
from sdl_orchestration.experiment.base_task import BaseTask, TaskStatus

from sdl_orchestration import sdl_config


class IncubatorTask(BaseTask):
    """
    This is the task for incubator.
    """

    def __init__(
        self,
        task_id,
        precursor_task_id: List[Optional[ObjectId]] | None = None,
        task_status: TaskStatus = TaskStatus.WAITING,
        priority: int = 0,
        samples: List[Optional[BaseSample]] | None = None,
        experiment_id: Optional[ObjectId] = None,
        targets: List[Optional[BaseSample]] | None = None,
        *args,
        **kwargs,
    ):

        if precursor_task_id is None:
            precursor_task_id = []

        if samples is None:
            samples = []

        super().__init__(
            task_id=task_id,
            precursor_task_id=precursor_task_id,
            task_status=task_status,
            priority=priority,
            samples=samples,
            experiment_id=experiment_id,
            targets=targets,
            *args,
            **kwargs,
        )

        self.task_dao = TaskDAO(self.task_id)

        # wait time in seconds
        self.wait_time_secs = sdl_config.incubator_wait_time_secs
        self.wait_time_hrs = sdl_config.incubator_wait_time_hrs

        self.targets = targets
        self.task_dao.create_entry(
            object_name=str(self.__class__.__name__),
            status=self.task_status,
            experiment_id=self.experiment_id,
            samples=self.samples,
            wait_time_hrs=self.wait_time_hrs,
            targets=self.targets,
        )

        self.timer = None

        logger.info(
            f"Creating incubator task: {task_id}, "
            f"wait time: {self.wait_time_hrs} hrs"
        )

    def run(self, *args, **kwargs) -> Any:
        """
        This run sets initiate timer for the incubator if called the first time.
        """
        if self.timer is None:
            self.timer = time.time()
            logger.info(f"Starting incubator task: {self.task_id}")
            self._update_status(status=TaskStatus.RUNNING)
            return self.task_id
        else:
            if self._check_time():
                self._after_run()
                return self.task_id

    def _update_status(self, status: TaskStatus):
        self.task_status = status
        self.task_dao.update_status(status=status)

    def _after_run(self):
        """
        This method is called after the incubator task has run for the
        specified wait time. It will update the status of the task to
        complete.
        """
        self._update_status(status=TaskStatus.COMPLETED)
        logger.info(f"Incubator finished running for {self.wait_time_hrs} hrs.")

    def check_condition(self, completed_task_list: List[ObjectId]):
        """
        This is a check condition method. This method checks if the
         task is ready to run.

         If the task is not ready to run, it will return False.
        """
        if self.timer is None:
            self._update_status(status=TaskStatus.WAITING)
            for task_id in self.precursor_task_id:
                if task_id not in completed_task_list:
                    logger.debug(
                        f"Task {self.task_id} is waiting for task "
                        f"{task_id} to complete."
                    )
                    return False
            self._update_status(status=TaskStatus.RUNNING)
            self.timer = time.time()  # start the timer
            logger.info(f"Timer set: {self.timer}")
            return True
        else:
            return self._check_time()

    def _check_time(self):
        """
        This method checks if the incubator task has run for the
        specified wait time. If the wait time has passed, it will
        call the after run method and return True.
        """
        if time.time() - self.timer > self.wait_time_secs:
            logger.info(
                f"Incubator task {self.task_id} has run for the "
                f"specified wait time of {self.wait_time_hrs} hrs."
            )
            return True
        human_readable_time = time.strftime(
            "%H:%M:%S", time.gmtime(self.wait_time_secs - (time.time() - self.timer))
        )
        logger.info(
            f"Incubator task {self.task_id} is still running."
            f" Remaining time: {human_readable_time}"
        )
        return False
