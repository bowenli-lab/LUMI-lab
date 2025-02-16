import time
from typing import Any, List, Optional

from bson.objectid import ObjectId

from sdl_orchestration import logger
from sdl_orchestration.database.daos import TaskDAO
from sdl_orchestration.experiment.base_sample import BaseSample
from sdl_orchestration.experiment.base_task import BaseTask, TaskStatus
from sdl_orchestration.communication.device_registry import device_registry


class RobotBaseTask(BaseTask):
    """
    This is the base task for robot.
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
        reagents: List[Optional[BaseSample]] | None = None,
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

        self.robot_controller = device_registry.robot_controller

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
        self.robot_controller.connect()
        logger.info(f"Robot connected.")

    def _after_run(self):
        """
        This is an after run method.
        """
        self._update_status(TaskStatus.COMPLETED)
        self.robot_controller.disconnect()
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

        if not self.robot_controller.is_running():
            return True
        else:
            logger.debug(f"Robot is still running. Task {self.task_id} " f"is waiting.")
            return False


# ==============================================================================
# Lipid synthesis tasks
# ==============================================================================
class RobotRetrivesReagentPlate4LipidSynthesis(RobotBaseTask):
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
            task_id,
            precursor_task_id,
            task_status,
            priority,
            samples,
            experiment_id,
        )
        self.liquid_sampler_controller = device_registry.liquid_sampler

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Lipid synthesis task: robot is retrieving the reagent " f"plate")
        self._before_run()

        # self.liquid_sampler_controller.run("plate_out")
        auto_clamp_callback = device_registry.clamp.auto_clamp()
        self.robot_controller.run(
            "s1_1_transfer_reagent_plate", callback=auto_clamp_callback
        )

        self._after_run()

    def check_condition(self, completed_task_list: List[ObjectId]):
        """
        TODO: make it a general method.

        This method is used to check the condition of the task, tailored for
        the lipid synthesis task.

        The robot can only pick up the tip racks when the previous task is
        completed and the opentron#0 is not running.
        """
        res = super().check_condition(completed_task_list)
        is_opentron_0_running = device_registry.opentron0.is_running()

        if res and not is_opentron_0_running:
            return True
        else:
            return False


class RobotPlacesReagentPlate4LipidSynthesis(RobotBaseTask):
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
            task_id,
            precursor_task_id,
            task_status,
            priority,
            samples,
            experiment_id,
        )
        self.liquid_sampler_controller = device_registry.liquid_sampler

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Lipid synthesis task: robot is placing the reagent " f"plate")
        self._before_run()

        self.robot_controller.run("s1_2_pickup_reagent_plate")

        self._after_run()


class RobotTransfersLNPPlate4LNPFormation(RobotBaseTask):
    def run(self, *args, **kwargs) -> Any:
        logger.info(f"LNP formation task: robot is transferring the LNP plate")
        self._before_run()

        clamp_callback = device_registry.clamp.auto_clamp()
        self.robot_controller.run("s2_1_transfer_lipid", callback=clamp_callback)

        self._after_run()


class RobotDisposeLipidPlate4LNPFormation(RobotBaseTask):
    def run(self, *args, **kwargs) -> Any:
        logger.info(f"LNP formation task: robot is disposing the lipid plate")
        self._before_run()

        self.robot_controller.run("s2_5_dispose_lipid_plate")

        self._after_run()


class TestClampTest4LNPFormation(RobotBaseTask):
    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Test task ")
        self._before_run()

        clamp_callback = device_registry.clamp.auto_clamp()
        self.robot_controller.run(
            "s1_1_transfer_reagent_plate", callback=clamp_callback
        )

        self._after_run()


class RobotAcquireLNPPlate4LNPFormation(RobotBaseTask):
    def run(self, *args, **kwargs) -> Any:
        logger.info(f"LNP formation task: robot is acquiring the LNP plate")
        self._before_run()

        device_registry.feeder2.feed_plate_by_type("PCR_DEEPWELL", 1)
        time.sleep(15)
        auto_clamp_callback = device_registry.clamp.auto_clamp()
        self.robot_controller.run("s2_2_pickup_lnp_plate", callback=auto_clamp_callback)

        self._after_run()


class RobotSealAquaEthanolPlate4LNPFormation(RobotBaseTask):
    def run(self, *args, **kwargs) -> Any:
        logger.info(f"LNP formation task: robot is sealing the aqua ethanol " f"plate")
        self._before_run()

        self.robot_controller.run("s2_3_seal_mat_aqua_ethanol")

        self._after_run()


class RobotAcquireTipRacks4LNPFormation(RobotBaseTask):
    def run(self, *args, **kwargs) -> Any:
        logger.info(f"LNP formation task: robot is acquiring the tip racks")
        self._before_run()

        feeder_callback = device_registry.feeder1.feed_plate_for_callback(3)
        self.robot_controller.run("s2_0_pickup3_racks", callback=feeder_callback)

        self._after_run()


class RobotTransfer2Shaker(RobotBaseTask):
    """
    This task is used to transfer the plate to the shaker.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"robot transfer 2 shaker")
        self._before_run()
        self.robot_controller.run("s1_4_transfer_to_shaker")

        self._after_run()


class RootSealReagentPlate4LipidSynthesis(RobotBaseTask):
    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Lipid synthesis task: robot is sealing the plate")
        self._before_run()

        self.robot_controller.run("s1_5_seal_reagent_plate")

        self._after_run()


class RobotAcquireLNPPlate4LipidSynthesis(RobotBaseTask):

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Lipid synthesis task: robot is acquiring the LNP plate")
        self._before_run()

        device_registry.feeder2.feed_plate_by_type("DEEPWELL_W_COVER", 1)
        time.sleep(15)
        clamp_callback = device_registry.clamp.auto_clamp()
        self.robot_controller.run("s1_3_pickup_lipid_plate", callback=clamp_callback)

        self._after_run()


class RobotDisposeTipRacks4LipidSynthesis(RobotBaseTask):
    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Lipid synthesis task: robot is disposing the tip racks")
        self._before_run()

        self.robot_controller.run("s1_6_dispose4_racks")

        self._after_run()


# ==============================================================================
# Cell dosing tasks
# =============================================================================


class RobotAcquireTipRacks4LipidSynthesis(RobotBaseTask):
    """
    The robot is moving the tip racks to the lipid synthesis station.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Lipid synthesis task: robot is picking up the tip racks")
        self._before_run()

        feeder_callback = device_registry.feeder0.feed_plate_for_callback(1)
        self.robot_controller.run("s1_0_pickup4_racks", callback=feeder_callback)

        self._after_run()

    def check_condition(self, completed_task_list: List[ObjectId]):
        """
        TODO: make it a general method.

        This method is used to check the condition of the task, tailored for
        the lipid synthesis task.

        The robot can only pick up the tip racks when the previous task is
        completed and the opentron#0 is not running.
        """
        res = super().check_condition(completed_task_list)
        is_opentron_0_running = device_registry.opentron0.is_running()

        if res and not is_opentron_0_running:
            return True
        else:
            return False


class RobotAcquireTipRacks4CellDosing(RobotBaseTask):
    """
    The robot is moving the tip racks to the cell dosing station.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Cell dosing task: robot is picking up the tip racks")
        self._before_run()

        feeder_callback = device_registry.feeder1.feed_plate_for_callback(1)
        self.robot_controller.run("s3_0_pickup1_racks", callback=feeder_callback)

        self._after_run()


class RobotTransferStack4CellDosing(RobotBaseTask):
    """
    The robot is transferring the stack to the cell dosing station.
    """

    def __init__(self, route: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route = route

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Cell dosing task: robot is transferring the stack")
        self._before_run()

        incubator_controller = device_registry.incubator
        incubator_controller.open_incubator()

        if self.route == 0:
            self.robot_controller.run("s3_1_transfer_to_stack_A")
        elif self.route == 1:
            self.robot_controller.run("s3_1_transfer_to_stack_B")

        incubator_controller.close_incubator()
        self._after_run()


class RobotTransferStack4PerpareReading(RobotBaseTask):
    """
    The robot is transferring the stack to the cell dosing station.
    """

    def __init__(self, route: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route = route

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Prepare reading: robot is transferring the stack")
        self._before_run()

        incubator_controller = device_registry.incubator
        incubator_controller.open_incubator()

        if self.route == 0:
            self.robot_controller.run("s4_2_transfer_to_stack_A")
        elif self.route == 1:
            self.robot_controller.run("s4_2_transfer_to_stack_B")

        incubator_controller.close_incubator()
        self._after_run()


class RobotAcquireEmpty4CellPlate4CellDosing(RobotBaseTask):
    """
    The robot is moving the cell plate to the cell dosing station.

    This task comes with two routes.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Cell plate task: robot is picking up the Cell plate")
        self._before_run()
        self.robot_controller.run("s3_2_pickup_cellplate")

        self._after_run()


class RobotTransferCellPlateToIncubator4CellDosing(RobotBaseTask):
    """
    The robot is transferring the cell plate to the incubator.
    """

    def __init__(self, route: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route = route

    def run(self, *args, **kwargs) -> Any:
        logger.info(
            f"Cell dosing task: robot is transferring the cell plate "
            f"to the incubator"
        )
        self._before_run()

        incubator_controller = device_registry.incubator

        incubator_controller.open_incubator()

        if self.route == 0:
            self.robot_controller.run("s3_3_transfer_cellplate_A")
        elif self.route == 1:
            self.robot_controller.run("s3_3_transfer_cellplate_B")
        else:
            incubator_controller.close_incubator()
            raise ValueError("Invalid route")

        incubator_controller.close_incubator()

        self._after_run()


class RobotDisposeTipRacks4CellDosing(RobotBaseTask):
    """
    The robot is disposing the tip racks after cell dosing.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Cell dosing task: robot is disposing the tip racks")
        self._before_run()

        self.robot_controller.run("s3_4_dispose1_racks")

        self._after_run()


class RobotDisposeLNPPlate4CellDosing(RobotBaseTask):
    """
    The robot is disposing the LNP plate after cell dosing.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Cell dosing task: robot is disposing the LNP plate")
        self._before_run()

        self.robot_controller.run("s3_5_dispose_lnp_plate")

        self._after_run()


class RobotDisposeTipRacks4LNPFormation(RobotBaseTask):
    """
    The robot is disposing the tip racks after LNP formation.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"LNP formation task: robot is disposing the tip racks")
        self._before_run()

        self.robot_controller.run("s2_4_dispose3_racks")

        self._after_run()


# ==============================================================================
# Preparing reading tasks
# ==============================================================================


class RobotAcquireTipRacks4PrepareReading(RobotBaseTask):
    """
    The robot is moving the tip racks to the opentron#1.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Prepare reading task: robot is moving the tip racks")
        self._before_run()

        feeder_callback = device_registry.feeder0.feed_plate_for_callback(1)
        self.robot_controller.run("s4_0_pickup1_racks", callback=feeder_callback)

        self._after_run()


class RobotAcquireReadingPlate4PrepareReading(RobotBaseTask):
    """
    The robot is moving the reading plate to the opentron#1.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Prepare reading task: robot is moving the reading plate")
        self._before_run()

        feeder_callback = device_registry.feeder3.feed_plate_for_callback(1)
        self.robot_controller.run("s4_1_pickup_white_plate", callback=feeder_callback)

        self._after_run()


class RobotOpenOneGlowMat4PrepareReading(RobotBaseTask):
    """
    The robot is opening one glow mat for reading.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Prepare reading task: robot is opening one glow mat")
        self._before_run()

        clamp_callback = device_registry.clamp.auto_clamp()
        self.robot_controller.run("s4_3_open_mat_oneglow", callback=clamp_callback)

        self._after_run()


class RobotSealOneGlowMat4PrepareReading(RobotBaseTask):
    """
    The robot is sealing one glow mat after reading preparation.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Prepare reading task: robot is sealing one glow mat")
        self._before_run()

        self.robot_controller.run("s4_9_seal_mat_oneglow")

        self._after_run()


class RobotTransferCellPlate4PrepareReading(RobotBaseTask):
    """
    The robot is transferring the cell plate to the opentron#1 from incubator.
    """

    def __init__(self, route: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route = route

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Prepare reading task: robot is transferring the cell plate")
        self._before_run()

        self.robot_controller.run("s4_4_pickup_cellplate")
        self._after_run()


class RobotDisposeTipRacks4PrepareReading(RobotBaseTask):
    """
    The robot is disposing the tip racks after reading preparation.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Prepare reading task: robot is disposing the tip racks")
        self._before_run()

        self.robot_controller.run("s4_5_dispose1_racks")
        self._after_run()


class RobotDisposeCellPlate4PrepareReading(RobotBaseTask):
    """
    The robot is disposing the cell plate after reading preparation.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Prepare reading task: robot is disposing the cell plate")
        self._before_run()

        self.robot_controller.run("s4_6_dispose_cell_plate")
        self._after_run()


# ==============================================================================
# Reading tasks
# ==============================================================================


class RobotSendCellPlate4Reading(RobotBaseTask):
    """
    The robot is sending the cell plate to the plate reader.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Reading task: robot is sending the cell plate")
        self._before_run()

        plate_reader_controller = device_registry.plate_reader
        plate_reader_controller.carrier_out()

        self.robot_controller.run("s4_7_to_plate_reader")

        plate_reader_controller.carrier_in()

        self._after_run()


class RobotDisposeCellPlate4Reading(RobotBaseTask):
    """
    The robot is disposing the cell plate after reading.
    """

    def run(self, *args, **kwargs) -> Any:
        logger.info(f"Reading task: robot is disposing the cell plate")
        self._before_run()

        plate_reader_controller = device_registry.plate_reader
        # plate_reader_controller.carrier_out()
        self.robot_controller.run("s4_8_dispose_plate")
        plate_reader_controller.carrier_in()
        self._after_run()


# ==============================================================================
# robot demo tasks
# ==============================================================================
