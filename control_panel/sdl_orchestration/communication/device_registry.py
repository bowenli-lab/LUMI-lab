from typing import Callable, List

from .devices import (
    OpentronController,
    RobotController,
    IncubatorController,
    PlateReaderController,
    LiquidSamplerController,
    FeederController,
)
from .devices.clamp_controller import ClampController
from .. import logger


class DeviceRegistry:
    """
    This class is used to register devices that are connected to the system.
    """

    robot: RobotController
    opentron0: OpentronController
    opentron1: OpentronController
    incubator: IncubatorController
    plate_reader: PlateReaderController
    liquid_sampler: LiquidSamplerController

    def __init__(self):
        self.robot_controller = RobotController(device_name="robot_ur5e")
        self.opentron0 = OpentronController(device_name="opentron0",
                                            device_code=0)
        self.opentron1 = OpentronController(device_name="opentron1",
                                            device_code=1)
        self.incubator = IncubatorController(device_name="incubator")
        self.plate_reader = PlateReaderController(device_name="plate_reader")
        self.liquid_sampler = LiquidSamplerController(device_name=
                                                      "liquid_sampler")
        self.feeder0 = FeederController(device_name="feeder0", device_code=0)
        self.feeder1 = FeederController(device_name="feeder1", device_code=1)
        self.feeder2 = FeederController(device_name="feeder2", device_code=2)
        self.feeder3 = FeederController(device_name="feeder3", device_code=3)
        self.clamp = ClampController(device_name="clamp")

    def get_devices(self) -> List:
        """
        This method is used to get the devices.
        """
        return [
            self.robot,
            self.opentron0,
            self.opentron1,
            self.incubator,
            self.plate_reader,
            self.liquid_sampler,
            self.feeder0,
            self.feeder1,
            self.feeder2,
            self.feeder3,
        ]

    def test_connection(self):
        """
        This method is used to test the connection to the devices.
        """
        pass

    def stop(self):
        """
        This method is used to stop the devices in case of an emergency.
        """
        # TODO: apply to all devices
        # try each
        self._stop(self.robot_controller)
        self._stop(self.opentron0)
        self._stop(self.opentron1)

    @staticmethod
    def _stop(device):
        """
        This method is used to stop the devices in case of an emergency.
        """
        try:
            device.stop()
        except Exception as e:
            logger.error(f"Device {device.device_name} failed to stop. "
                         f"Error: {e}")


device_registry = DeviceRegistry()
