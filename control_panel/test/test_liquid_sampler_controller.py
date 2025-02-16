import sys
import time
from unittest import TestCase

from bson import ObjectId

sys.path.insert(0, "../")


class TestSamplerController(TestCase):

    @classmethod
    def setUpClass(self):
        super(TestSamplerController, self).setUpClass()
        from sdl_orchestration.communication.devices.liquid_sampler_controller import (
            LiquidSamplerController,
        )

        self.liquid_sampler = LiquidSamplerController()

    # def test0(self):

    # def run_exp(well_lst):
    #     N = 200
    #     for well in well_lst:
    #         self.liquid_sampler.pump_by_well(well=well, volume=N)

    # self.liquid_sampler.plate_in()
    # run_exp(["H4",])
    # self.liquid_sampler.plate_out()

    # # self.liquid_sampler.pump_by_well("A2", 1)
    # pass

    def test1(self):
        # for alphabet in "H":
        #     for number in range(6, 13):
        #         well = alphabet + str(number)
        #         self.liquid_sampler.pump_by_well(well, 2)
        #         time.sleep(1)
        pass

    # double check B3, D7, E11, F2, F5, G10, H4

    def test2(self):
        pass
        # reagents_to_be_pumped = ['r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r19']
        # volumes_to_be_pumped = [1, 2, 4, 2, 1, 2, 3]
        # self.liquid_sampler.sample_reagents(reagents_to_be_pumped,
        #                                     volumes_to_be_pumped)

    def test3(self):
        self.liquid_sampler.plate_in()
        wells_to_be_pumped = ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"]
        volumes_to_be_pumped = [
            100,
        ] * 8
        # wells_to_be_pumped = [alphabet + str(number) for alphabet in "ABCD" for number in range(1, 13)]
        # volumes_to_be_pumped = [1] * 48
        self.liquid_sampler.pump_batch_by_well(wells_to_be_pumped, volumes_to_be_pumped)
        self.liquid_sampler.plate_out()
        pass
