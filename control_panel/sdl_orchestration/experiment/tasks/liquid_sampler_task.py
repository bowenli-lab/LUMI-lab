import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from bson.objectid import ObjectId
from sdl_orchestration import device_registry, logger, sdl_config
from sdl_orchestration.database.daos import TaskDAO
from sdl_orchestration.experiment import sample_registry, StockUsageAndRefill
from sdl_orchestration.experiment.base_sample import BaseSample
from sdl_orchestration.experiment.base_task import BaseTask, TaskStatus
from sdl_orchestration.experiment.samples.plate96well import Plate96Well
from sdl_orchestration.experiment.samples.sample_registry import sample_registry


class LiquidSamplerBaseTask(BaseTask):
    """
    This is the base task for liquid sampler.
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
        self.task_dao.create_entry(
            object_name=str(self.__class__.__name__),
            status=self.task_status,
            experiment_id=self.experiment_id,
            samples=self.samples,
        )

        self.liquid_sampler_controller = device_registry.liquid_sampler

        logger.info(f"Creating liquid sampler task: {task_id}")

    def run(self, *args, **kwargs) -> Any:
        """
        This is a run method. It is a placeholder method.
        """
        raise NotImplementedError

    def _after_run(
        self,
    ):
        """
        This is an after run method.

        This method updates the task status to completed and disconnects the
        opentron controller.
        """
        self._update_status(TaskStatus.COMPLETED)

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
        self._update_status(TaskStatus.WAITING)

        for task_id in self.precursor_task_id:
            if task_id not in completed_task_list:
                logger.debug(
                    f"Task {self.task_id} is waiting for task "
                    f"{task_id} to complete."
                )
                return False

        self._update_status(TaskStatus.READY)

        if not self.liquid_sampler_controller.is_running():
            return True
        else:
            logger.debug(
                f"Opentron {self.liquid_sampler_controller} is"
                f" still running. Task {self.task_id} is waiting."
            )
            return False


class LiquidSampling(LiquidSamplerBaseTask):
    """
    This is the opentron prepare reading task.
    """

    # fmt: off
    ALL_96_WELLS = [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12",
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12",
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12",
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
        "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12",
        "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12",
    ]
    ALL_WELLS_TUBE_OFFSET = {
        "A1": 500, "A2": 500, "A3": 500, "A4": 500, "A5": 500, "A6": 500, "A7": 500, "A8": 500, "A9": 500, "A10": 500, "A11": 500, "A12": 500,
        "B1": 500, "B2": 500, "B3": 500, "B4": 800, "B5": 500, "B6": 500, "B7": 500, "B8": 900, "B9": 500, "B10": 500, "B11": 500, "B12": 500,
        "C1": 500, "C2": 500, "C3": 500, "C4": 800, "C5": 500, "C6": 500, "C7": 500, "C8": 500, "C9": 500, "C10": 500, "C11": 600, "C12": 500,
        "D1": 550, "D2": 500, "D3": 500, "D4": 500, "D5": 500, "D6": 600, "D7": 600, "D8": 500, "D9": 500, "D10": 500, "D11": 600, "D12": 500,
        "E1": 550, "E2": 500, "E3": 500, "E4": 800, "E5": 500, "E6": 500, "E7": 500, "E8": 500, "E9": 500, "E10": 500, "E11": 600, "E12": 600,
        "F1": 500, "F2": 500, "F3": 500, "F4": 500, "F5": 500, "F6": 500, "F7": 500, "F8": 500, "F9": 500, "F10": 500, "F11": 500, "F12": 500,
        "G1": 500, "G2": 500, "G3": 500, "G4": 500, "G5": 500, "G6": 500, "G7": 500, "G8": 500, "G9": 500, "G10": 500, "G11": 500, "G12": 500,
        "H1": 500, "H2": 500, "H3": 500, "H4": 500, "H5": 500, "H6": 500, "H7": 500, "H8": 500, "H9": 500, "H10": 500, "H11": 500, "H12": 500,
    }
    # fmt: on

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Args:
            stock_usage_refill (StockUsageAndRefill): The stock usage and refill, both are dictionaries of the form: {well_str: volume}, e.g. {"A1": 100}.
        """
        assert "stock_usage_refill" in kwargs, "stock_usage_refill is required"
        self.stock_usage_refill: StockUsageAndRefill = kwargs.pop("stock_usage_refill")
        super().__init__(*args, **kwargs)
        self.update_dao = kwargs.get("update_dao", True)
        logger.info(f"Whether to update dao in sampling: {self.update_dao}")

    @staticmethod
    def _map_reagent(targets: Plate96Well) -> Tuple[List[str], List[str]]:
        """
        This method computes the reagent for the samples and map them to the
        reagent plate.

        Returns:
            Two lists of strings: The first list is the reagent plate location
            of the lipid_components, the second list is the lipid structure of the
            lipid_components.
        """
        return_list = []
        lipid_str_list = []
        reagent_plate = sample_registry.reagent
        for lipids in targets.get_all_lipids():
            amines, isocyanide, lipid_carboxylic_acid, lipid_aldehyde = (
                lipids.get_lipid_structures()
            )
            for lipid_component in [
                amines,
                isocyanide,
                lipid_carboxylic_acid,
                lipid_aldehyde,
            ]:
                if lipid_component.is_empty():
                    # we pass the blank for control samples
                    continue
                lipid_str_list.append(str(lipid_component))
                res = reagent_plate.locate_reagent(lipid_component)
                if res is None:
                    logger.error(f"Cannot find reagent {lipid_component} for {lipids}")
                    raise Exception(
                        f"Cannot find reagent {lipid_component} for {lipids}"
                    )
                else:
                    row, col = res
                    alpha_num = reagent_plate.coordinate2alphabet(row, col)
                    return_list.append(alpha_num)
        return return_list, lipid_str_list

    def run(self, *args, **kwargs) -> Any:
        """
        This method runs the liquid sampling task, it samples the reagents
        based on the target lipids.
        """
        logger.info(f"Running liquid sampling task: {self.task_id}")
        usage = self.stock_usage_refill.usage
        refill = self.stock_usage_refill.refill

        # REFILL THE REAGENT BY CALLING THE API AND UPDATE DATABASE
        if len(refill) > 0:
            # Plate in
            # self.liquid_sampler_controller.run("plate_in")

            wells_to_refill = refill.keys()
            volumes_to_refill = [refill[well] for well in wells_to_refill]
            logger.info(
                f"Refilling {wells_to_refill} \n with volumes: {volumes_to_refill}"
            )

            if self.update_dao:
                # TODO: also need to sync dao after the liquid synthesis task
                sample_registry.reagent.fill_targets_by_alphanum_well(
                    wells_to_refill, volumes_to_refill
                )

            # PRE-pump step to pump 500ul of reagent out
            wells_refill_offsets = [
                self.ALL_WELLS_TUBE_OFFSET[well] for well in wells_to_refill
            ]
            # self.liquid_sampler_controller.pump_batch_by_well(
            #     wells_to_refill, wells_refill_offsets
            # )
            # # actual pump
            # self.liquid_sampler_controller.pump_batch_by_well(
            #     wells_to_refill, volumes_to_refill
            # )
            # # POST-pump step to aspirate 400ul of reagent in
            # self.liquid_sampler_controller.aspirate_batch_by_well(
            #     wells_to_refill, wells_refill_offsets
            # )

            # Plate out
            # self.liquid_sampler_controller.run("plate_out")
        else:
            logger.info("Skipping refill step, all reagents are sufficient.")

        # VIRTUAL USE THE REAGENT BY CALLING THE API AND UPDATE DATABASE
        wells_to_use = usage.keys()
        volumes_to_use = [usage[well] for well in wells_to_use]
        if self.update_dao:
            sample_registry.reagent.sample_targets_by_alphanum_well(
                wells_to_use, volumes_to_use
            )

        self._after_run()
