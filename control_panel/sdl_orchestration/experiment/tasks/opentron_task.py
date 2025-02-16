import time
from typing import Any, List, Optional

from bson.objectid import ObjectId

from sdl_orchestration import logger, device_registry, sdl_config
from sdl_orchestration.database.daos import OpentronDAO, TaskDAO
from sdl_orchestration.experiment.base_sample import BaseSample
from sdl_orchestration.experiment.base_task import BaseTask, TaskStatus

from sdl_orchestration.communication.devices.opentron_controller import (
    OpentronController,
    OpentronException,
)
from sdl_orchestration.experiment.handlers.protocal_handler import ProtocolFactory
from sdl_orchestration.experiment.samples.plate96well import Plate96Well
from sdl_orchestration.utils.task_util import get_full_path_for_file
from sdl_orchestration.utils import parse_human_readable_time


class UseOpentronN:
    """
    This is an abstract class for identifying which opentron controller to use.
    """

    current_controller = None


class UseOpentron0(UseOpentronN):
    """
    This is a class for using opentron0.
    """

    current_controller = device_registry.opentron0


class UseOpentron1(UseOpentronN):
    """
    This is a class for using opentron1.
    """

    current_controller = device_registry.opentron1


class OpentronBaseTask(BaseTask, UseOpentronN):
    """
    This is the base task for opentron.
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
        self.task_dao.create_entry(
            object_name=str(self.__class__.__name__),
            status=self.task_status,
            experiment_id=self.experiment_id,
            samples=self.samples,
            targets=targets,
        )

        logger.info(f"Creating opentron task: {task_id}")

    def run(self, *args, **kwargs) -> Any:
        """
        This is a run method. It is a placeholder method.
        """
        raise NotImplementedError

    def _after_run(self, controller: OpentronController):
        """
        This is an after run method.

        This method updates the task status to completed and disconnects the
        opentron controller.
        """
        self._update_status(TaskStatus.COMPLETED)
        controller.disconnect()
        logger.info(f"{controller.device_name} disconnected.")

    def _update_status(self, status: TaskStatus):
        """
        This is an update status method.
        """
        self.task_status = status
        self.task_dao.update_status(status=status)

    def check_condition(self, completed_task_list: List[ObjectId]):
        """
        This is a check condition method; it checks if the precursor tasks
        have been completed.
        """
        if self.task_status == TaskStatus.RUNNING:
            return True
        if not hasattr(self, "current_controller"):
            raise AttributeError("current_controller is not defined.")
        current_controller = self.current_controller

        if self.task_status not in [
            TaskStatus.WAITING,
            TaskStatus.READY,
            TaskStatus.RUNNING,
        ]:
            self._update_status(TaskStatus.WAITING)

        for task_id in self.precursor_task_id:
            if task_id not in completed_task_list:
                logger.debug(
                    f"Task {self.task_id} is waiting for task "
                    f"{task_id} to complete."
                )
                return False

        if self.task_status == TaskStatus.WAITING:
            self._update_status(TaskStatus.READY)

        if not current_controller.is_running():
            # the controller is idle
            return True
        else:
            # the controller is still running
            logger.debug(
                f"Opentron {current_controller.device_name} is"
                f" still running. Task {self.task_id} is waiting."
            )
            return False


class OpentronCellDosing(OpentronBaseTask, UseOpentron1):
    """
    This is the opentron cell dosing task.
    """

    def __init__(
        self,
        task_id,
        precursor_task_id: List[Optional[ObjectId]] | None = None,
        task_status: TaskStatus = TaskStatus.WAITING,
        priority: int = 0,
        samples: List[Optional[BaseSample]] | None = None,
        experiment_id: Optional[ObjectId] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            task_id=task_id,
            precursor_task_id=precursor_task_id,
            task_status=task_status,
            priority=priority,
            samples=samples,
            experiment_id=experiment_id,
            device_code=1,
            *args,
            **kwargs,
        )

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the cell dosing task.
        """

        current_controller = self.current_controller
        current_controller.connect()
        current_controller.run(
            "cell_dosing",
            experiment_id=self.experiment_id,
        )
        self._after_run(current_controller)


class OpentronCellDosingDemo(OpentronBaseTask, UseOpentron1):
    """
    This is the opentron cell dosing task.
    """

    def __init__(
        self,
        task_id,
        precursor_task_id: List[Optional[ObjectId]] | None = None,
        task_status: TaskStatus = TaskStatus.WAITING,
        priority: int = 0,
        samples: List[Optional[BaseSample]] | None = None,
        experiment_id: Optional[ObjectId] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            task_id=task_id,
            precursor_task_id=precursor_task_id,
            task_status=task_status,
            priority=priority,
            samples=samples,
            experiment_id=experiment_id,
            device_code=1,
            *args,
            **kwargs,
        )

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the cell dosing task.
        """
        current_controller = self.current_controller
        current_controller.connect()
        current_controller.run(
            "demo_cell_dosing",
            experiment_id=self.experiment_id,
        )
        self._after_run(current_controller)


class OpentronPrepareReading(OpentronBaseTask, UseOpentron1):
    """
    This is the opentron prepare reading task.
    """

    def __init__(self, route: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route = route

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the prepare reading task.
        """
        current_controller = self.current_controller
        current_controller.connect()
        if self.route == 0:
            current_controller.run(
                "transfer_cell_to_wells_before_reading_A",
                experiment_id=self.experiment_id,
            )
        elif self.route == 1:
            current_controller.run(
                "transfer_cell_to_wells_before_reading_B",
                experiment_id=self.experiment_id,
            )

        logger.info("One glo added; Starting reading.")
        # logger.info("One glo added; Waiting for 3 minutes.")
        # time.sleep(180)
        # logger.info("3 minutes passed; Starting reading.")
        self._after_run(current_controller)


class OpentronPrepareReadingDemo(OpentronBaseTask, UseOpentron1):
    """
    This is the opentron prepare reading task.
    """

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the prepare reading task.
        """
        current_controller = self.current_controller
        current_controller.connect()
        current_controller.run(
            "demo_transfer_cell_to_wells_before_reading",
            experiment_id=self.experiment_id,
        )
        self._after_run(current_controller)


class OpentronLipidSynthesis(OpentronBaseTask, UseOpentron0):
    """
    This is the opentron lipid synthesis task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the lipid synthesis task.
        """
        current_controller = self.current_controller
        current_controller.connect()

        # we parse the synthesis protocol here
        parsed_protocol = self._parse_protocol("lipid_synthesis_template")
        current_controller.run(
            parsed_protocol,
            experiment_id=self.experiment_id,
        )

        self._after_run(current_controller)

    def _parse_protocol(self, program: str) -> str:
        """
        This method parses the protocol file's content and returns
        the parsed protocol's file name.

        Args:
            program (str): The (modified, if any) program file name.
        Returns:
            str: The parsed protocol file name.
        """
        protocol_factory = ProtocolFactory()
        persist_protocol_dir = sdl_config.protocol_configs["parsed_protocol_path"]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        parsed_protocol = program + "_" + timestamp + ".py"
        persisted_protocol_path = f"{persist_protocol_dir}/{parsed_protocol}"
        parse_source = get_full_path_for_file(
            program, sdl_config.protocol_configs["program_path"]
        )

        protocol_factory.get_protocal(
            protocol_name=program,
            parse_source=parse_source,
            parse_dest=persisted_protocol_path,
            experiment_id=self.experiment_id,
            targets=self.targets,
        )
        return parsed_protocol


class OpentronDemoShaker(OpentronBaseTask, UseOpentron0):
    """
    This is the opentron prepare reading task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the prepare reading task.
        """
        current_controller = self.current_controller
        current_controller.run(
            "demo_shaker",
            experiment_id=self.experiment_id,
        )
        self._after_run(current_controller)


class OpentronLNPFormation(OpentronBaseTask, UseOpentron1):
    """
    This is the opentron lnp formation task.
    """

    def __init__(self, route: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route = route

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the lnp formation task.
        """
        current_controller = self.current_controller
        if self.route == 0:
            current_controller.run(
                "lnp_formation_A",
                experiment_id=self.experiment_id,
            )
        elif self.route == 1:
            current_controller.run(
                "lnp_formation_B",
                experiment_id=self.experiment_id)
        self._after_run(current_controller)


class OpentronLNPFormationDemo(OpentronBaseTask, UseOpentron1):
    """
    This is the opentron lnp formation task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the lnp formation task.
        """
        current_controller = self.current_controller
        current_controller.run(
            "demo_lnp_formation",
            experiment_id=self.experiment_id,
        )
        self._after_run(current_controller)


class OpentronDemoSynthesis(
    OpentronBaseTask,
    UseOpentron0,
):
    """
    This is the opentron prepare reading task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the prepare reading task.
        """
        current_controller = self.current_controller
        current_controller.connect()
        current_controller.run(
            "demo_lipid_synthesis",
            experiment_id=self.experiment_id,
        )
        self._after_run(current_controller)


class OpentronShaking(
    OpentronBaseTask,
    UseOpentron0,
):
    """
    This is the opentron shaking task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the shaking task.
        """
        current_controller = self.current_controller
        current_controller.connect()
        current_controller.run(
            "lipid_shaking",
            experiment_id=self.experiment_id,
        )
        self._after_run(current_controller)


class OpentronShakingAsync(
    OpentronBaseTask,
    UseOpentron0,
):
    """
    This is the opentron shaking task.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = None
        self.start_time = None

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the shaking task.
        """
        current_controller = self.current_controller
        current_controller.connect()
        if self.run_id is None:
            self._update_status(TaskStatus.RUNNING)
            self.run_id = current_controller.submit_run("lipid_shaking")
            self.start_time = time.time()
        else:
            # query the run status
            run_status = current_controller.query_run(self.run_id)
            if run_status == "succeeded":
                self.current_controller.send_notification(
                    task_name="lipid_shaking",
                    progress="completed",
                    experiment_id=self.experiment_id,
                )
                self._after_run(current_controller)
            elif run_status == "running" or run_status == "finishing":
                time_elapsed = time.time() - self.start_time
                logger.debug(
                    f"Run {self.run_id} is still running."
                    f"{parse_human_readable_time(int(time_elapsed))} "
                    f"has passed."
                )
                return
            else:
                logger.error(f"Run {self.run_id} errored, status: {run_status}")
                raise OpentronException("Run errored")
