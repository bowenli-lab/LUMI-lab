import time
from abc import ABC
from queue import Queue
from typing import Any, Dict, List, Optional
from bson import ObjectId
from sdl_orchestration import logger
from sdl_orchestration.communication.device_registry import device_registry
from sdl_orchestration.database.daos import ExperimentDAO
from sdl_orchestration.experiment import (
    BaseExperiment,
    BaseSample,
    ExperimentStatus,
    base_task,
    StockUsageAndRefill,
)
from sdl_orchestration.experiment.samples.plate96well import Plate96Well
from sdl_orchestration.experiment.tasks.incubator_task import IncubatorTask
from sdl_orchestration.experiment.tasks.liquid_sampler_task import LiquidSampling
from sdl_orchestration.experiment.tasks.opentron_task import (
    OpentronCellDosing,
    OpentronCellDosingDemo,
    OpentronDemoShaker,
    OpentronDemoSynthesis,
    OpentronLNPFormation,
    OpentronLNPFormationDemo,
    OpentronLipidSynthesis,
    OpentronPrepareReading,
    OpentronPrepareReadingDemo,
)
from sdl_orchestration.experiment.tasks.opentron_task import (
    OpentronShakingAsync as OpentronShaking,
)
from sdl_orchestration.experiment.tasks.plate_reader_task import PlateReaderDoReading
from sdl_orchestration.experiment.tasks.robot_task import (
    RobotAcquireEmpty4CellPlate4CellDosing,
    RobotAcquireLNPPlate4LNPFormation,
    RobotAcquireLNPPlate4LipidSynthesis,
    RobotAcquireReadingPlate4PrepareReading,
    RobotAcquireTipRacks4CellDosing,
    RobotAcquireTipRacks4LNPFormation,
    RobotAcquireTipRacks4LipidSynthesis,
    RobotAcquireTipRacks4PrepareReading,
    RobotDisposeCellPlate4PrepareReading,
    RobotDisposeCellPlate4Reading,
    RobotDisposeLNPPlate4CellDosing,
    RobotDisposeLipidPlate4LNPFormation,
    RobotDisposeTipRacks4CellDosing,
    RobotDisposeTipRacks4LNPFormation,
    RobotDisposeTipRacks4LipidSynthesis,
    RobotDisposeTipRacks4PrepareReading,
    RobotOpenOneGlowMat4PrepareReading,
    RobotPlacesReagentPlate4LipidSynthesis,
    RobotRetrivesReagentPlate4LipidSynthesis,
    RobotSealAquaEthanolPlate4LNPFormation,
    RobotSealOneGlowMat4PrepareReading,
    RobotSendCellPlate4Reading,
    RobotTransfer2Shaker,
    RobotTransferCellPlate4PrepareReading,
    RobotTransferCellPlateToIncubator4CellDosing,
    RobotTransferStack4CellDosing,
    RobotTransferStack4PerpareReading,
    RobotTransfersLNPPlate4LNPFormation,
    RootSealReagentPlate4LipidSynthesis,
    TestClampTest4LNPFormation,
)


class ProductionExperiment(BaseExperiment):
    """
    ProductionExperiment is a class that represents a production-ready
    experiment. It is a subclass of BaseExperiment.
    """

    def __init__(
        self,
        experiment_id: Optional[ObjectId] | str = None,
        targets: Optional[Plate96Well] = None,
        reagents: Optional[BaseSample] = None,
        experiment_index: Optional[int] = None,
        stock_usage_refill: Optional[StockUsageAndRefill] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        This is the constructor of the ProductionExperiment class.

        Args:
            experiment_id (Optional[ObjectId], optional): The id of the experiment. Defaults to None.
            targets (Optional[Plate96Well], optional): The targets, e.g. lipids to be sythesised.
            reagents (Optional[BaseSample], optional): The reagents of the experiment. Not used for now.
            experiment_index (Optional[int], optional): The index of the experiment. Defaults to None.
            stock_usage_refill (`StockUsageAndRefill`, optional): The reagents to use and to be refilled. Defaults to None.

            *args: Variable length argument list.

        """

        if experiment_id is None:
            experiment_id = ObjectId()

        super().__init__(experiment_id, targets, reagents, *args, **kwargs)

        self.task_queue = Queue()
        self.completed_tasks = []
        self.running_tasks = []
        self.route = experiment_index % 2
        self.stock_usage_refill = stock_usage_refill
        logger.info(
            f"Experiment {experiment_index} <{experiment_id}> selects: route {self.route}"
        )

        self.tasks = self.compile_task()
        for task in self.tasks:
            self.task_queue.put(task)

        """
        Route 0 is for the A experiment and Route 1 is for the B experiment.
        """

        self.experiment_dao = ExperimentDAO(self.experiment_id)
        self.experiment_dao.create_entry(
            object_name="",
            status=self.status,
            targets=self.targets,
            experiment_index=experiment_index,
        )

    def compile_task(self) -> List[base_task.BaseTask]:
        """
        This method compiles the tasks for the experiment.
        """
        return self._compile_task()

    def stop(self):
        """
        This method stops the experiment.
        """
        logger.critical("Stopping the experiment...")
        device_registry.stop()
        self.status = ExperimentStatus.INTERRUPTED

    def _compile_task(self) -> List[base_task.BaseTask]:
        """
        This method compiles the tasks for the experiment.
        """
        tasks = []
        tasks += self._compile_synthesis_tasks(precursor_task_id=[])
        tasks += self._compile_lnp_formation_tasks(
            precursor_task_id=[tasks[-1].task_id] if tasks else []
        )
        tasks += self._compile_cell_dosing_tasks(
            precursor_task_id=[tasks[-1].task_id] if tasks else []
        )
        tasks += self._compile_prepare_reading(
            precursor_task_id=[tasks[-1].task_id] if tasks else []
        )
        # tasks += self._compile_test_tasks()
        return tasks

    def persist_state(self):
        state = {
            "completed_tasks": self.completed_tasks,
            "running_tasks": self.running_tasks,
            "task_queue": list(self.task_queue.queue),
            "status": self.status,
        }
        self.experiment_dao.update_state(state)

    def load_state(self):
        state = self.experiment_dao.get_state()
        if state:
            self.completed_tasks = state.get("completed_tasks", [])
            self.running_tasks = state.get("running_tasks", [])
            self.task_queue.queue = state.get("task_queue", Queue()).queue
            self.status = state.get("status", ExperimentStatus.CREATED)

    def _compile_test_tasks(self) -> List[base_task.BaseTask]:

        test_task = OpentronLipidSynthesis(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            reagents=self.sample,
        )

        return [test_task]

    def _compile_synthesis_tasks(
        self,
        precursor_task_id: List[ObjectId],
    ) -> List[base_task.BaseTask]:
        """
        This method compiles the synthesis tasks for the experiment.
        """
        s1_get_reagent = RobotRetrivesReagentPlate4LipidSynthesis(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=precursor_task_id,
            reagents=self.sample,
        )

        s1_sample_reagent = LiquidSampling(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_get_reagent.task_id],
            reagents=self.sample,
            stock_usage_refill=self.stock_usage_refill,
        )

        s1_robot_get_filled_reagent_plate = RobotPlacesReagentPlate4LipidSynthesis(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_sample_reagent.task_id],
            reagents=self.sample,
        )

        s1_get_tip_racks = RobotAcquireTipRacks4LipidSynthesis(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_robot_get_filled_reagent_plate.task_id],
            reagents=self.sample,
        )

        s1_opentron_get_lnp = RobotAcquireLNPPlate4LipidSynthesis(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_get_tip_racks.task_id],
            reagents=self.sample,
        )

        s1_opentron_do_synthesis = OpentronLipidSynthesis(
            # s1_opentron_do_synthesis = OpentronDemoSynthesis(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_opentron_get_lnp.task_id],
            reagents=self.sample,
        )

        s1_robot_2_shaker = RobotTransfer2Shaker(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_opentron_do_synthesis.task_id],
            reagents=self.sample,
        )

        s1_seal_plate = RootSealReagentPlate4LipidSynthesis(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_robot_2_shaker.task_id],
            reagents=self.sample,
        )

        s1_opentron_shake = OpentronShaking(
            # s1_opentron_shake = OpentronDemoShaker(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_seal_plate.task_id],
            reagents=self.sample,
        )

        s1_dispose_tip_racks = RobotDisposeTipRacks4LipidSynthesis(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_opentron_shake.task_id],
            reagents=self.sample,
        )

        return [
            s1_get_reagent,
            s1_sample_reagent,
            s1_robot_get_filled_reagent_plate,
            s1_get_tip_racks,
            s1_opentron_get_lnp,
            s1_opentron_do_synthesis,
            s1_robot_2_shaker,
            s1_seal_plate,
            s1_opentron_shake,
            s1_dispose_tip_racks,
        ]

    def _compile_lnp_formation_tasks(
        self,
        precursor_task_id: List[ObjectId],
    ) -> List[base_task.BaseTask]:
        """
        This method compiles the lnp formation tasks for the experiment.
        """

        s2_get_robot_tip_rack = RobotAcquireTipRacks4LNPFormation(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=precursor_task_id,
            reagents=self.sample,
        )

        s2_get_lipid = RobotTransfersLNPPlate4LNPFormation(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s2_get_robot_tip_rack.task_id],
            reagents=self.sample,
        )

        s2_get_lnp_plate = RobotAcquireLNPPlate4LNPFormation(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s2_get_lipid.task_id],
            reagents=self.sample,
        )

        s2_make_lnp = OpentronLNPFormation(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s2_get_lnp_plate.task_id],
            reagents=self.sample,
            route=self.route,
        )

        s2_seal_aqua_ethanol = RobotSealAquaEthanolPlate4LNPFormation(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s2_make_lnp.task_id],
            reagents=self.sample,
        )

        s2_dispose_tip_racks = RobotDisposeTipRacks4LNPFormation(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s2_seal_aqua_ethanol.task_id],
            reagents=self.sample,
        )

        s2_dispose_lnp_plate = RobotDisposeLipidPlate4LNPFormation(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s2_dispose_tip_racks.task_id],
            reagents=self.sample,
        )

        return [
            s2_get_robot_tip_rack,
            s2_get_lipid,
            s2_get_lnp_plate,
            s2_make_lnp,
            s2_seal_aqua_ethanol,
            s2_dispose_tip_racks,
            s2_dispose_lnp_plate,
        ]

    def _compile_cell_dosing_tasks(
        self,
        precursor_task_id: List[ObjectId],
    ) -> List[base_task.BaseTask]:
        """
        This method compiles the cell dosing tasks for the experiment.
        """

        s3_first_plate_get_tip_racks = RobotAcquireTipRacks4CellDosing(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=precursor_task_id,
            reagents=self.sample,
        )

        s3_two_plates_get_empty_cell_plate = RobotTransferStack4CellDosing(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_first_plate_get_tip_racks.task_id],
            reagents=self.sample,
            route=self.route,
        )

        s3_first_plate_get_cell_plate = RobotAcquireEmpty4CellPlate4CellDosing(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_two_plates_get_empty_cell_plate.task_id],
            reagents=self.sample,
            route=self.route,
        )

        s3_first_plate_opentron_dose_cell = OpentronCellDosing(
            # s3_first_plate_opentron_dose_cell = OpentronCellDosingDemo(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_first_plate_get_cell_plate.task_id],
            reagents=self.sample,
        )

        s3_first_plate_transfer_to_incubator = (
            RobotTransferCellPlateToIncubator4CellDosing(
                task_id=ObjectId(),
                experiment_id=self.experiment_id,
                targets=self.targets,
                precursor_task_id=[s3_first_plate_opentron_dose_cell.task_id],
                reagents=self.sample,
                route=self.route,
            )
        )

        s3_first_plate_dispose_tip_racks = RobotDisposeTipRacks4CellDosing(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_first_plate_transfer_to_incubator.task_id],
            reagents=self.sample,
        )

        s3_second_plate_get_tip_racks = RobotAcquireTipRacks4CellDosing(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_first_plate_dispose_tip_racks.task_id],
            reagents=self.sample,
        )

        s3_second_plate_get_cell_plate = RobotAcquireEmpty4CellPlate4CellDosing(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_second_plate_get_tip_racks.task_id],
            reagents=self.sample,
        )

        s3_second_plate_opentron_dose_cell = OpentronCellDosing(
            # s3_second_plate_opentron_dose_cell = OpentronCellDosingDemo(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_second_plate_get_cell_plate.task_id],
            reagents=self.sample,
        )

        s3_second_plate_transfer_to_incubator = (
            RobotTransferCellPlateToIncubator4CellDosing(
                task_id=ObjectId(),
                experiment_id=self.experiment_id,
                targets=self.targets,
                precursor_task_id=[s3_second_plate_opentron_dose_cell.task_id],
                reagents=self.sample,
                route=self.route,
            )
        )

        s3_second_plate_dispose_tip_racks = RobotDisposeTipRacks4CellDosing(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_second_plate_transfer_to_incubator.task_id],
            reagents=self.sample,
        )

        s3_dispose_lnp_plate = RobotDisposeLNPPlate4CellDosing(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_second_plate_dispose_tip_racks.task_id],
            reagents=self.sample,
        )

        incubator_task = IncubatorTask(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s3_dispose_lnp_plate.task_id],
            reagents=self.sample,
        )

        return [
            s3_first_plate_get_tip_racks,
            s3_two_plates_get_empty_cell_plate,
            s3_first_plate_get_cell_plate,
            s3_first_plate_opentron_dose_cell,
            s3_first_plate_transfer_to_incubator,
            s3_first_plate_dispose_tip_racks,
            s3_second_plate_get_tip_racks,
            s3_second_plate_get_cell_plate,
            s3_second_plate_opentron_dose_cell,
            s3_second_plate_transfer_to_incubator,
            s3_second_plate_dispose_tip_racks,
            s3_dispose_lnp_plate,
            incubator_task,
        ]

    def _compile_prepare_reading(
        self,
        precursor_task_id: List[ObjectId],
    ) -> List[base_task.BaseTask]:
        s4_first_plate_get_tip_racks = RobotAcquireTipRacks4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=precursor_task_id,
            reagents=self.sample,
        )

        s4_first_plate_get_reading_plate = RobotAcquireReadingPlate4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_get_tip_racks.task_id],
            reagents=self.sample,
        )

        s4_two_plates_transfer_from_incubator = RobotTransferStack4PerpareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_get_reading_plate.task_id],
            reagents=self.sample,
            route=self.route,
        )

        s4_first_plate_open_one_glow_mat = RobotOpenOneGlowMat4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_two_plates_transfer_from_incubator.task_id],
            reagents=self.sample,
        )

        s4_first_plate_get_cell_plate = RobotTransferCellPlate4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_open_one_glow_mat.task_id],
            reagents=self.sample,
            route=self.route,
        )

        s4_first_plate_opentron_prepare_reading = OpentronPrepareReading(
            # s4_first_plate_opentron_prepare_reading = OpentronPrepareReadingDemo(
            route=self.route,
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_get_cell_plate.task_id],
            reagents=self.sample,
        )

        s4_first_plate_dispose_tip_racks = RobotDisposeTipRacks4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_opentron_prepare_reading.task_id],
            reagents=self.sample,
        )

        s4_first_plate_dispose_cell_plates = RobotDisposeCellPlate4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_dispose_tip_racks.task_id],
            reagents=self.sample,
        )

        s4_first_plate_to_plate_reader = RobotSendCellPlate4Reading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_dispose_cell_plates.task_id],
            reagents=self.sample,
            route=self.route,
        )

        s1_first_plate_do_reading = PlateReaderDoReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_to_plate_reader.task_id],
            reagents=self.sample,
        )

        s4_first_plate_dispose_reading_plate = RobotDisposeCellPlate4Reading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s1_first_plate_do_reading.task_id],
            reagents=self.sample,
        )

        s4_second_plate_get_tip_racks = RobotAcquireTipRacks4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_first_plate_dispose_reading_plate.task_id],
            reagents=self.sample,
        )

        s4_second_plate_get_reading_plate = RobotAcquireReadingPlate4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_get_tip_racks.task_id],
            reagents=self.sample,
        )

        s4_second_plate_get_cell_plate = RobotTransferCellPlate4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_get_reading_plate.task_id],
            reagents=self.sample,
            route=self.route,
        )

        s4_second_plate_opentron_prepare_reading = OpentronPrepareReading(
            route=self.route,
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_get_cell_plate.task_id],
            reagents=self.sample,
        )

        s4_seal_one_glow_mat = RobotSealOneGlowMat4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_opentron_prepare_reading.task_id],
            reagents=self.sample,
        )

        s4_second_plate_dispose_tip_racks = RobotDisposeTipRacks4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_opentron_prepare_reading.task_id],
            reagents=self.sample,
        )

        s4_second_plate_dispose_cell_plates = RobotDisposeCellPlate4PrepareReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_dispose_tip_racks.task_id],
            reagents=self.sample,
        )

        s4_second_plate_to_plate_reader = RobotSendCellPlate4Reading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_dispose_cell_plates.task_id],
            reagents=self.sample,
        )

        s4_second_plate_do_reading = PlateReaderDoReading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_to_plate_reader.task_id],
            reagents=self.sample,
        )

        s4_second_plate_dispose_reading_plate = RobotDisposeCellPlate4Reading(
            task_id=ObjectId(),
            experiment_id=self.experiment_id,
            targets=self.targets,
            precursor_task_id=[s4_second_plate_do_reading.task_id],
            reagents=self.sample,
        )

        return [
            s4_first_plate_get_tip_racks,
            s4_first_plate_get_reading_plate,
            s4_two_plates_transfer_from_incubator,
            s4_first_plate_open_one_glow_mat,
            s4_first_plate_get_cell_plate,
            s4_first_plate_opentron_prepare_reading,
            s4_first_plate_dispose_tip_racks,
            s4_first_plate_dispose_cell_plates,
            s4_first_plate_to_plate_reader,
            s1_first_plate_do_reading,
            s4_first_plate_dispose_reading_plate,
            s4_second_plate_get_tip_racks,
            s4_second_plate_get_reading_plate,
            s4_second_plate_get_cell_plate,
            s4_second_plate_opentron_prepare_reading,
            s4_seal_one_glow_mat,
            s4_second_plate_dispose_tip_racks,
            s4_second_plate_dispose_cell_plates,
            s4_second_plate_to_plate_reader,
            s4_second_plate_do_reading,
            s4_second_plate_dispose_reading_plate,
        ]
