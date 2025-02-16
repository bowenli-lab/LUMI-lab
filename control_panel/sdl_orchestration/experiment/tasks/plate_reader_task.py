import time
from typing import Any, List, Optional

from bson.objectid import ObjectId

from sdl_orchestration import logger, sdl_config
from sdl_orchestration.database.daos import TaskDAO
from sdl_orchestration.experiment.base_sample import BaseSample
from sdl_orchestration.experiment.base_task import BaseTask, TaskStatus
from sdl_orchestration.communication.device_registry import device_registry
from sdl_orchestration.experiment.samples.plate96well import Plate96Well


class PlaterReaderBaseTask(BaseTask):
    """
    This is the base task for plate reader.
    """

    def __init__(
        self,
        task_id,
        precursor_task_id: List[Optional[ObjectId]] | None = None,
        task_status: TaskStatus = TaskStatus.WAITING,
        priority: int = 0,
        samples: List[Optional[BaseSample]] | None = None,
        experiment_id: Optional[ObjectId] = None,
        targets: Optional[Plate96Well] | None = None,
        reagents: List[Optional[Plate96Well]] | None = None,
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
            reagents=reagents,
            *args,
            **kwargs,
        )

        self.task_dao = TaskDAO(self.task_id)
        self.targets = targets
        self.reagents = reagents
        self.task_dao.create_entry(
            object_name=str(self.__class__.__name__),
            status=self.task_status,
            experiment_id=self.experiment_id,
            samples=self.samples,
            targets=self.targets,
            reagents=self.reagents,
        )

        self.plate_reader_controller = device_registry.plate_reader

        logger.info(f"Creating robot task: {task_id}")

    def run(self, *args, **kwargs) -> Any:
        """
        This is a test run method.
        """
        raise NotImplementedError

    def _before_run(self):
        """
        This is a before run method.
        """
        self._update_status(TaskStatus.RUNNING)
        self.plate_reader_controller.connect()
        logger.info(f"Robot connected.")

    def _after_run(self):
        """
        This is an after run method.
        """
        self._update_status(TaskStatus.COMPLETED)
        self.plate_reader_controller.disconnect()
        logger.info(f"Robot disconnected.")

    def _update_status(self, status: TaskStatus):
        """
        This is an update status method.
        """
        self.task_status = status
        self.task_dao.update_status(status=status)

    def check_condition(self, completed_task_list: List[ObjectId]):
        """
        This is a check condition method.
        """
        self.task_dao.update_status(status=TaskStatus.WAITING)

        for task_id in self.precursor_task_id:
            if task_id not in completed_task_list:
                logger.debug(
                    f"Task {self.task_id} is waiting for task "
                    f"{task_id} to complete."
                )
                return False

        self.task_dao.update_status(status=TaskStatus.READY)

        if not self.plate_reader_controller.is_running():
            return True
        else:
            logger.debug(
                f"Plate reader is still running. Task {self.task_id} " f"is waiting."
            )
            return False


class PlateReaderDoReading(PlaterReaderBaseTask):
    """
    This is the plate reader do reading task.
    """

    def run(self, *args, **kwargs) -> Any:
        """
        This is a run method.
        """
        self._before_run()

        logger.info(f"Running plate reader do reading task: {self.task_id}")

        protocol_path = sdl_config.protocol_path
        time_stamp = time.strftime("%m%d-%H%M%S")
        targets_identifier = self.targets.get_identifier()

        for i in range(2):

            # the plate reading will be done twice, the index of replicate
            # will be recorded as the ending index in the protocol path

            round_identifier = f"{i + 1}"

            reading_path = (
                sdl_config.reading_output_base
                + f"/{self.experiment_id}-{targets_identifier}-{time_stamp}-{round_identifier}.csv"
            )
            experiment_id = (
                sdl_config.experimental_output_base
                + f"/{self.experiment_id}-{targets_identifier}-{time_stamp}-{round_identifier}.xpt"
            )

            logger.info(f"Protocol path: {protocol_path}")

            self.plate_reader_controller.read_plate(
                protocol_path=protocol_path,
                csv_path=reading_path,
                experiment_path=experiment_id,
            )

            reading_info, results = self.plate_reader_controller.parse_plate(
                csv_path=reading_path
            )

            self.targets.add_reading_info(reading_info, results)

        self._after_run()
