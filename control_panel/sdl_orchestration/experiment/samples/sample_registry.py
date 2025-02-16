from typing import Dict, List

from sdl_orchestration.experiment.samples.plate96well import Plate96Well


class SampleRegistry:
    """
    This class is used to register devices that are connected to the system.
    """

    reagent: Plate96Well
    sample_collection: Dict[str, Plate96Well]

    def __init__(self):
        self.reagent = Plate96Well(is_reagent=True)
        self.sample_collection = {}


sample_registry = SampleRegistry()
