import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue
from typing import Any, List, Optional

from bson import ObjectId

from . import base_task
from .base_sample import BaseSample
from .base_task import BaseTask
from .. import logger


class ExperimentStatus(Enum):
    STOPPED = "STOPPED"
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"
    INTERRUPTED = "INTERRUPTED"


class BaseExperiment(ABC):
    """
    This is the base class for all experiments. All experiments should
    inherit from this class.
    """

    targets: Optional[Any] = []
    experiment_id: Optional[ObjectId] = None
    sample: Optional[BaseSample] = None

    def __init__(
        self,
        experiment_id: Optional[ObjectId] = None,
        targets: Any = None,
        reagents: Optional[BaseSample] = None,
        experiment_index: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        This is the constructor of the BaseExperiment class.

        Args:

        experiment_id (Optional[ObjectId], optional): The id of the
        experiment. Defaults to None.
        targets (Any, optional): The targets of the experiment. Defaults to None.
        reagents (Optional[BaseSample], optional): The reagents of the
        experiment. Defaults to None.
        """
        self.targets = targets
        self.sample = reagents
        self.experiment_id = experiment_id

        self.task_queue = Queue()
        self.status = ExperimentStatus.CREATED

        self.completed_tasks = []
        self.running_tasks = []
        self.failed_tasks = []

        self.pause_event = threading.Event()
        self.pause_event.set()

        self.experiment_dao = None

    @abstractmethod
    def compile_task(self) -> List[BaseTask]:
        """
        This method returns a list of tasks that will be executed by the
        workers. This method should be implemented by the child class. It
        should contain the logic of the experiment.
        """
        raise NotImplementedError

    def run(self):
        """
        This method runs the experiment.

        1. Prepare to run the experiment.
        2. It iterates over the tasks in the task queue until the queue is
        empty, or there is no runnable task.
        3. If there is no runnable task, it switches to waiting mode.
        """
        self._prepare_to_run()
        initial_queue_size = self.task_queue.qsize()
        non_runnable_count = 0

        while not self.task_queue.empty():
            self.pause_event.wait()  # Wait if paused

            task = self.task_queue.get()
            if task.check_condition(self.completed_tasks):
                # check if condition of task is met
                try:
                    task.run()
                except KeyboardInterrupt:
                    logger.error(f"Task {task.task_id} was interrupted.")
                    task.fail("Task was interrupted.")
                    self._fail()
                    self.stop()
                    break
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed with error: {e}")
                    task_name = task.__class__.__name__
                    error_message = (
                        f"Task {task_name}<{task.task_id}> " f"failed with error: {e}"
                    )
                    task.fail(error_message)
                    self.failed_tasks.append(task.task_id)
                    self._fail()
                    break

                if task.task_status == base_task.TaskStatus.COMPLETED:
                    logger.info("completed task")
                    # verify if the task is completed
                    self.completed_tasks.append(task.task_id)
                    if task.task_id in self.running_tasks:
                        # remove the task from running list
                        self.running_tasks.remove(task.task_id)
                elif task.task_status == base_task.TaskStatus.RUNNING:
                    # non-blocking task, requeue the task and add
                    # to running list
                    logger.debug("requeuing task")
                    self.task_queue.put(task)
                    if task.task_id not in self.running_tasks:
                        self.running_tasks.append(task.task_id)
                    non_runnable_count += 1
            else:
                # requeue the task if condition is not met
                non_runnable_count += 1
                # logger.debug()
                self.task_queue.put(task)
            # break condition

            if non_runnable_count == initial_queue_size and initial_queue_size != 0:
                logger.debug(
                    f"Experiment {self.experiment_id} is stuck. "
                    f"Switching to waiting mode."
                )
                break  # Break the loop if no tasks were runnable
            logger.debug(
                f"Experiment {self.experiment_id} is running. Checking" f" next task..."
            )

            time.sleep(1)
        if not self.task_queue.empty():
            self._update_status(ExperimentStatus.RUNNING)
        else:
            self._finish()

    def _prepare_to_run(self):
        """
        This method prepares the experiment to run.
        """
        self._update_status(ExperimentStatus.RUNNING)
        logger.info(f"Experiment {self.experiment_id} is ready to run")

    def pause(self):
        self.status = ExperimentStatus.PAUSED
        logger.info(f"Experiment {self.experiment_id} is paused.")
        self.pause_event.clear()

    def stop(self):
        self.status = ExperimentStatus.STOPPED
        self.pause_event.clear()
        self._fail_all_ongoing_tasks()

    def resume(self):
        self.status = ExperimentStatus.RUNNING
        self.pause_event.set()

    def _finish(self):
        """
        This method finishes the experiment.
        """
        if self.status == ExperimentStatus.FAILED:
            logger.error(f"Experiment {self.experiment_id} failed")
        else:
            self._update_status(ExperimentStatus.COMPLETED)
            logger.info(f"Experiment {self.experiment_id} is finished")

    def _fail(self):
        """
        This method fails the experiment and fail all
        the tasks in the experiment
        """
        self._update_status(ExperimentStatus.FAILED)
        self._fail_all_ongoing_tasks()
        logger.error(f"Experiment {self.experiment_id} failed")
        raise Exception(f"Experiment {self.experiment_id} failed")

    def _fail_all_ongoing_tasks(self):
        """
        This method fails all the ongoing tasks in the experiment.
        """
        for task_id in self.running_tasks:
            task = self.task_queue.get(task_id)
            task.auto_fail()
            self.completed_tasks.append(task_id)
        logger.error(f"Experiment {self.experiment_id} triggered auto-fail. ")

    def retry_failed_tasks(self):
        """
        This method retries all failed tasks in the experiment.
        """
        for task_id in self.failed_tasks:
            task = self.task_queue.get(task_id)
            if task:
                self.task_queue.put(task)
                logger.info(
                    f"Retrying task {task_id} in experiment {self.experiment_id}"
                )
        self.failed_tasks = []

    def _update_status(self, status: ExperimentStatus):
        """
        This method updates the status of the experiment.

        Args:
            status (ExperimentStatus): The new status of the experiment.
        """
        self.status = status
        self.experiment_dao.update_status(status)

    def __repr__(self):
        return f"Experiment<{self.experiment_id}>"
