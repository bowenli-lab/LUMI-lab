import time
from queue import Queue
from typing import Dict

from bson import ObjectId

from sdl_orchestration.logger import logger
from .base_experiment import BaseExperiment, ExperimentStatus
import threading


class ExperimentManager:
    """
    ExperimentManager is a class that manages the experiments. It is
    responsible for creating, running, and monitoring the experiments. It is
    also responsible for saving the results of the experiments.
    """

    def __init__(self):
        self.experiment_queue = Queue()
        self.experiments: Dict[str, BaseExperiment] = {}
        self.running = False

    def run(self):
        """
        This method runs the experiment in the queue.
        """
        self.running = True
        logger.info("Starting running the experiments...")
        while self.running:
            if not self.experiment_queue.empty():
                self._loop()
            time.sleep(2)
        logger.info("All experiments finished. " "Experiment manager is shutting down.")

    def _loop(self):
        """
        This method is the main loop of the experiment manager.
        It checks the queue for new experiments and runs them.
        """
        if not self.experiment_queue.empty():
            experiment = self.experiment_queue.get()
            logger.debug(f"Running experiment: {experiment}")

            # Run the experiment
            experiment.run()

            if experiment.status == ExperimentStatus.COMPLETED:
                logger.info(f"Experiment Manager: Experiment finished: {experiment}")
            elif experiment.status == ExperimentStatus.RUNNING:
                # requeue the experiment
                self.experiment_queue.put(experiment)
                logger.debug(f"Experiment Manager: Experiment re-queued: {experiment}")
            elif experiment.status == ExperimentStatus.PAUSED:
                logger.info(f"Experiment Manager: Experiment paused: {experiment}")
            else:
                logger.error(f"Experiment Manager: Experiment failed: {experiment}")

    def propose_experiment(self, experiment: BaseExperiment):
        """This method proposes an experiment to the experiment manager."""
        self.experiment_queue.put(experiment)
        self.experiments[str(experiment.experiment_id)] = experiment
        logger.info(f"Experiment proposed: {experiment.experiment_id}")

    def start(self):
        self.running = True
        thread = threading.Thread(target=self.run)
        thread.start()

    def resume(self, experiment_id: str):
        experiment = self.experiments.get(str(experiment_id))
        if experiment:
            logger.info(f"Resuming experiment: {experiment_id}")
            experiment.resume()

    def pause(self, experiment_id: str):
        experiment = self.experiments.get(str(experiment_id))
        if experiment:
            logger.info(f"Pausing experiment: {experiment_id}")
            experiment.pause()

    def stop(self):
        self.running = False
        for experiment in self.experiments.values():
            experiment.stop()
        # clear queue
        while not self.experiment_queue.empty():
            self.experiment_queue.get()
