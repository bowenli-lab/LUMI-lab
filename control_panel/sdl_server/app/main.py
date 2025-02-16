import datetime
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
from dataclasses import field

from bson import ObjectId
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append("..")
from sdl_orchestration import logger, sdl_config
from sdl_orchestration.experiment import (
    LipidStructure,
    StockUsageAndRefill,
    sample_registry,
)
from sdl_orchestration.experiment.experiment_manager import ExperimentManager
from sdl_orchestration.experiment.experiments import ProductionExperiment
from sdl_orchestration.experiment.samples.plate96well import Plate96Well
from sdl_orchestration.experiment.tasks.liquid_sampler_task import LiquidSampling
from sdl_orchestration.utils.enum_lipid import (
    LipidStructureEnumerator,
    lipid_list2plate96well_lst,
)

app = FastAPI()

job_registry = {}
experiment_registry = {}
# a global experiment manager that may run exps of multiple jobs
experiment_manager = ExperimentManager()


class LipidComponents(BaseModel):
    """
    This class represents the components of a lipid.
    """

    amines: str
    isocyanide: str
    lipid_carboxylic_acid: str
    lipid_aldehyde: str


class TargetLipid(BaseModel):
    """
    This class represents a target lipid. Currently specified by the
    components of the lipid.
    """

    lipid: LipidComponents


class ExperimentInput(BaseModel):
    """
    This class represents the input for an experiment.
    Contains a list of target lipids.
    """

    targets: List[TargetLipid]


class JobStatus:
    """
    This class represents the status of an experiment.
    """

    proposed = "proposed"
    approved = "approved"
    started = "started"
    paused = "paused"
    resumed = "resumed"
    stopped = "stopped"
    retrying = "retrying"
    failed = "failed"
    completed = "completed"


class JobProposal:
    """
    This class represents a proposed experiment.

    Args:
        job_id (str): the ID of the job.
        experiment_input_dict (Dict[str, ExperimentInput]): a dictionary
        containing the experiment_id and its corresponding experiment input.
        status (str): the status of the job.
        approval (bool): whether the job has been approved
    """

    job_id: str = str(ObjectId())
    experiment_input_dict: Dict[str, ExperimentInput] = (
        {}
    )  # FIXME potential type conflict, to be fixed
    status: str = JobStatus.proposed
    approval: bool = False
    target_plates: Dict[str, Plate96Well] = None
    # usage_and_refills: List[StockUsageAndRefill]

    def __init__(
        self, job_id, experiment_input_dict, status=JobStatus.proposed, approval=False
    ):
        self.job_id = job_id
        self.experiment_input_dict = experiment_input_dict
        self.status = status
        self.approval = approval
        self.target_plates = {}
        for experiment_id, experiment_input in self.experiment_input_dict.items():
            # Convert target lipids into the plate format
            lipid_list = parse_experiment_input(experiment_input)
            target_plate = lipid_list2plate96well_lst(
                lipid_list, num_control=4, control_pos=["A1", "B1", "A12", "B12"]
            )[0]
            self.target_plates[experiment_id] = target_plate


class StockManager:
    """
    This class represents the stock manager. It controls how much to refill the
    stock in 96 Deep Well Plate, and record the stock volume of each lipid component
    at real time, sync with database.
    """

    MAX_VOLUME = sdl_config.liquid_sampler_safe_cap
    SAFE_VOLUME = sdl_config.liquid_sampler_safe_volume
    ALLOWED_VOLUME_USAGE = MAX_VOLUME - SAFE_VOLUME

    def __init__(self, job_to_manage: JobProposal, inplace=False):
        """
        Args:
            job_to_manage (JobProposal): the job to manage.
            inplace (bool, optional): whether to modify the job proposal info inplace.
                                    Defaults to False.

        """
        # init stock from database
        self.reagent_stock: Plate96Well = sample_registry.reagent
        self.job_to_manage = job_to_manage
        self.target_plates = job_to_manage.target_plates
        self.inplace = inplace

    def _calculate_needed_volume(
        self,
    ) -> List[Dict[str, float]]:
        """
        This function calculates the needed volume of each lipid component
        for the job, including all experiments in the job.

        Returns:
            Dict[str, float]: a dictionary containing the lipid component and
            its needed volume for all experiments in the job.
        """
        # total_needed_volume_dict = {}
        # calculate needed volume of each lipid component
        needed_volume_per_experiment = []
        for id, target_plate in self.target_plates.items():
            res, lipid_str_list = LiquidSampling._map_reagent(target_plate)
            lipid_counter = dict(Counter(lipid_str_list))
            well_counter = dict(Counter(res))
            logger.info(f"Lipid reagent usage: {lipid_counter} for experiment {id}")

            volumes_to_sample = {
                key: well_counter[key] * sdl_config.liquid_sampler_sample_volume
                for key in well_counter.keys()
            }
            logger.info(f"Well usage (ul): {volumes_to_sample} for experiment {id}")
            needed_volume_per_experiment.append(volumes_to_sample)

            # for well, volume in volumes_to_sample.items():
            #     if well not in total_needed_volume_dict:
            #         total_needed_volume_dict[well] = 0
            #     total_needed_volume_dict[well] += volume
        return needed_volume_per_experiment

    def _plan_refill_single_reagent(
        self, well: str, volume_per_exp: List[float]
    ) -> Tuple[float, float]:
        """
        This function plans the refill of a single reagent.

        Args:
            well (str): the reagent well name, which will be potentially refilled.
            volume_per_exp (List[float]): the volume needed for each experiment.

        Rules:
        1. Check the volume of the reagent in the well.
        2. Count the post-use volume (pv) by subtracting the volume_needed from the volume.
        3. If the post-use volume is less than :attr:`SAFE_VOLUME`, trigger refilling.
        Refill the well with :attr:`MAX_VOLUME`. This will be the first refill, whose
        volume is refill_1 = :attr:`MAX_VOLUME` - current_volume.
        4. If the post-use volume is still less than :attr:`SAFE_VOLUME`, trigger the second refill.

        """
        assert len(volume_per_exp) == 2, "Only support 2 experiments for now."
        assert (
            volume_per_exp[0] <= self.ALLOWED_VOLUME_USAGE
        ), f"Volume usage {volume_per_exp[0]} ul of well {well} in experiment 1 exceeds the allowed volume usage of {self.ALLOWED_VOLUME_USAGE} ul."
        assert (
            volume_per_exp[1] <= self.ALLOWED_VOLUME_USAGE
        ), f"Volume usage {volume_per_exp[1]} ul of well {well} in experiment 2 exceeds the allowed volume usage of {self.ALLOWED_VOLUME_USAGE} ul."

        refill_1 = None
        refill_2 = None

        total_volume_needed = sum(volume_per_exp)

        current_volume = self.reagent_stock.get_well_by_alphanum(well).volume
        post_use_volume = current_volume - total_volume_needed
        if post_use_volume < self.SAFE_VOLUME:  # trigger refill 1
            refill_1 = self.MAX_VOLUME - current_volume
            post_use_volume += refill_1
        if post_use_volume < self.SAFE_VOLUME:  # trigger refill 2
            # calculate the left volume after refill 1 and experiment 1
            left_volume = self.MAX_VOLUME - volume_per_exp[0]
            refill_2 = self.MAX_VOLUME - left_volume
        return refill_1, refill_2

    def plan_refill(self) -> Tuple[StockUsageAndRefill, StockUsageAndRefill]:
        """
        This function plans the refill of all reagents.
        """
        all_96_well_strs = LiquidSampling.ALL_96_WELLS
        volume_per_exp = self._calculate_needed_volume()
        volume_to_refill_1 = {}
        volume_to_refill_2 = {}
        for well in all_96_well_strs:
            this_well_usage_per_exp = [exp.get(well, 0) for exp in volume_per_exp]
            refill_1, refill_2 = self._plan_refill_single_reagent(
                well, this_well_usage_per_exp
            )
            if refill_1:
                volume_to_refill_1[well] = refill_1
            if refill_2:
                volume_to_refill_2[well] = refill_2

        usage_and_refill_1 = StockUsageAndRefill(
            usage=volume_per_exp[0], refill=volume_to_refill_1
        )
        usage_and_refill_2 = StockUsageAndRefill(
            usage=volume_per_exp[1], refill=volume_to_refill_2
        )

        if self.inplace:
            # update the stock volume in place
            self.job_to_manage.usage_and_refills = [
                usage_and_refill_1,
                usage_and_refill_2,
            ]

        return usage_and_refill_1, usage_and_refill_2


def parse_experiment_input(experiment_data: ExperimentInput):
    """
    This function parses the experiment input and returns a
    list of LipidStructure objects.

    Args:
        experiment_data (ExperimentInput): the input for the experiment.
    """
    lipid_list = [
        LipidStructure(
            amines=target.lipid.amines,
            isocyanide=target.lipid.isocyanide,
            lipid_carboxylic_acid=target.lipid.lipid_carboxylic_acid,
            lipid_aldehyde=target.lipid.lipid_aldehyde,
        )
        for target in experiment_data.targets
    ]
    return lipid_list


@app.post("/propose_job")
def propose_job(experiment_inputs: List[ExperimentInput]):
    """
    This function proposes a new job and register
    its corresponding experiments.

    Args:
        experiment_inputs (List[ExperimentInput]): the input for the
        proposed experiment.

    Returns:
        Dict: a dictionary containing the job_id, experiment_ids, and status.
    """
    # job_id = str(ObjectId())
    # # using datetime as the job_id
    job_id = f"job_{datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')}"

    experiment_input_dict = {}
    experiment_id_list = []
    for experiment_input in experiment_inputs:
        # register the proposed experiment
        experiment_id = str(ObjectId())
        experiment_input_dict[experiment_id] = experiment_input
        experiment_registry[experiment_id] = experiment_input
        experiment_id_list.append(experiment_id)

    # make a job proposal
    job_proposal = JobProposal(
        job_id=job_id, experiment_input_dict=experiment_input_dict
    )

    # calculate the refill for this job
    stock_manager = StockManager(job_proposal, inplace=True)
    usage_and_refills = stock_manager.plan_refill()

    logger.info(
        f"Refills for job <{job_id}> : {usage_and_refills[0].refill}, {usage_and_refills[1].refill}"
    )

    job_registry[job_id] = job_proposal

    return {
        "job_id": job_id,
        "experiment_ids": experiment_id_list,
        "status": JobStatus.proposed,
    }


@app.post("/approve_experiment/{job_id}")
def approve_job(job_id: str):
    """
    This function approves a proposed experiment.

    Args:
        job_id (str): the ID of the job to approve.
    """
    experiment_proposal = job_registry.get(job_id)
    if not experiment_proposal:
        raise HTTPException(status_code=404, detail="Experiment not found")
    experiment_proposal.approval = True
    experiment_proposal.status = JobStatus.approved
    return {"job_id": job_id, "status": JobStatus.approved}


@app.get("/proposed_jobs")
def get_proposed_jobs():
    """
    This function returns a list of all proposed experiments.
    """
    res = []
    for job in job_registry.values():
        res.append(
            {
                "job_id": job.job_id,
                "experiment_input_dict": job.experiment_input_dict,
                "usage_and_refills": [i.to_dict() for i in job.usage_and_refills],
                "status": job.status,
                "approval": job.approval,
            }
        )
    return res


@app.post("/start/{job_id}")
def start_jobs(job_id):
    """
    This function starts the specified job if the job has been approved.
    """
    # check if the job has been approved
    job_proposal: JobProposal = job_registry.get(job_id)
    if job_proposal is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # if not job_proposal.approval:
    #     raise HTTPException(status_code=400, detail="Job not approved")

    usage_and_refills = job_proposal.usage_and_refills

    # start the job
    experiment_ids = []
    try:
        for (
            experiment_id,
            experiment_input,
        ) in job_proposal.experiment_input_dict.items():
            target_plate = job_proposal.target_plates[experiment_id]
            experiment_index = len(experiment_ids) % 2
            experiment = ProductionExperiment(
                experiment_id=str(experiment_id),
                targets=target_plate,
                experiment_index=experiment_index,
                stock_usage_refill=usage_and_refills[experiment_index],
            )
            experiment_manager.propose_experiment(experiment)
            experiment_ids.append(experiment_id)
        experiment_manager.start()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "job_id": job_id,
        "experiment_ids": experiment_ids,
        "status": JobStatus.started,
    }


@app.get("/experiment_details/{experiment_id}")
def get_experiment_details(experiment_id: str = None):
    """
    This function returns the details of the experiment with the given
    experiment_id. If experiment_id is not provided, it returns the details
    of all experiments.

    Args:
        experiment_id (str, optional): the ID of the experiment. Defaults to None.
    """
    if experiment_id:
        experiment = experiment_manager.experiments.get(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="experiment not found")
        return experiment
    else:
        return list(experiment_manager.experiments.values())


@app.post("/pause/{experiment_id}")
def pause_experiment(experiment_id: str):
    """
    This function pauses the experiment with the given experiment_id.
    """
    experiment_manager.pause(experiment_id)
    return {"experiment_id": experiment_id, "status": "experiment paused"}


@app.post("/resume/{experiment_id}")
def resume_experiment(experiment_id: str):
    experiment_manager.resume(experiment_id)
    return {"experiment_id": experiment_id, "status": "experiment resumed"}


@app.get("/stop_all")
def stop_experiment():
    experiment_manager.stop()
    # list the experiment ids
    experiment_ids = experiment_manager.experiments.keys()
    return [
        {"experiment_id": experiment_id, "status": "experiment stopped"}
        for experiment_id in experiment_ids
    ]


@app.post("/retry/{experiment_id}")
def retry_experiment(experiment_id: str):
    experiment = experiment_manager.experiments.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="experiment not found")
    experiment.retry_failed_tasks()
    return {"experiment_id": experiment_id, "status": "retrying failed tasks"}


@app.get("/status/{experiment_id}")
def get_status(experiment_id: str):
    experiment = experiment_manager.experiments.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="experiment not found")
    return {"experiment_id": experiment_id, "status": experiment.status.name}
