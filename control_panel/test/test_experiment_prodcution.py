from unittest import TestCase

import sys
sys.path.insert(0, "../")

from sdl_orchestration import device_registry
from sdl_orchestration.experiment.experiment_manager import ExperimentManager
from sdl_orchestration.utils.enum_lipid import LipidStructureEnumerator, \
    lipid_list2plate96well_lst


class TestExperimentManager(TestCase):
    def setUp(self):
        from sdl_orchestration.experiment.experiments.production_ready_experiment import \
            ProductionExperiment
        self.experiment_manager = ExperimentManager()

        lipid_gen = LipidStructureEnumerator().get_generator()

        lipid_list = [i for i in lipid_gen]
        target_plates = lipid_list2plate96well_lst(lipid_list, 4)

        test_experiment1 = ProductionExperiment(targets=target_plates[0],
                                                experiment_index=0, )
        test_experiment2 = ProductionExperiment(targets=target_plates[0],
                                                experiment_index=1, )
        self.experiment_manager.propose_experiment(test_experiment1)
        self.experiment_manager.propose_experiment(test_experiment2)

    def test_run_experiment(self):
        self.experiment_manager.run()
        self.assertTrue(True)
