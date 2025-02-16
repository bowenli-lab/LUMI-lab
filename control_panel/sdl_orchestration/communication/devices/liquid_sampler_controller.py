import requests
from sdl_orchestration.communication.device_controller import \
    BaseDeviceController, DeviceStatus
import time
from typing import List, Optional
from bson import ObjectId
from sdl_orchestration import logger, sdl_config
from sdl_orchestration.database.daos import LiquidSamplerDAO
from sdl_orchestration.utils.liquid_sampler_calibration import \
    LiquidSamplerCalibration


def _map_reagents_to_motor_numbers(reagents: List[str]) -> List[int]:
    """
    This method maps reagents to motor numbers.
    """
    mapped_motor_numbers = []
    for reagent in reagents:
        mapped_motor_numbers.append(sdl_config.reagent2motor_mapping[reagent])
    return mapped_motor_numbers


def _map_well_to_motor_number(well: str) -> int:
    """
    This method maps well to motor number.
    """
    # validate the well string
    if well not in sdl_config.well2motor_mapping:
        raise ValueError(f"Well {well} not found in the mapping table.")
    return sdl_config.well2motor_mapping[well]


def _map_motor_number_to_well(motor_number: int) -> str:
    """
    This method maps motor number to well.
    """
    # validate the motor number
    if motor_number not in sdl_config.motor2well_mapping:
        raise ValueError(f"Motor number {motor_number} not found in the "
                         f"mapping table.")
    return sdl_config.motor2well_mapping[motor_number]


class LiquidSamplerController(BaseDeviceController):

    @staticmethod
    def _calibrate_volume_decorator(func):
        def wrapper(self, *args, **kwargs):
            logger.info(kwargs)
            if "volume" not in kwargs:
                raise ValueError("Volume not provided.")
            if "motor_number" not in kwargs:
                raise ValueError("Motor number not provided.")
            volume = kwargs["volume"]
            motor_number = kwargs["motor_number"]
            well = _map_motor_number_to_well(motor_number)
            unit_value = \
                self.calibration.calculate_units_for_volume(volume).loc[
                    self.calibration.calculate_units_for_volume(volume)[
                        "Well"] == well][
                    "Units Needed"].values[0]

            # sample the coefficient
            unit_value = (unit_value *
                          sdl_config.liquid_sampler_sample_coefficient)

            logger.info(f"Calibrating sampling to {unit_value} units. {volume}"
                        f"uL (with coefficient "
                        f"{sdl_config.liquid_sampler_sample_coefficient})")
            kwargs["volume"] = unit_value
            return func(self, **kwargs, )

        return wrapper

    @staticmethod
    def _calibrate_volumes_decorator(func):
        def wrapper(self, *args, **kwargs):
            logger.info(kwargs)
            if "volumes" not in kwargs:
                raise ValueError("Volumes not provided.")
            if "motor_numbers" not in kwargs:
                raise ValueError("Motor numbers not provided.")
            volumes = kwargs["volumes"]
            motor_numbers = kwargs["motor_numbers"]
            # wells = [_map_motor_number_to_well(motor_number) for motor_number in motor_numbers]
            unit_values = []
            for volume, motor_number in zip(volumes, motor_numbers):
                well = _map_motor_number_to_well(motor_number)
                unit_value = \
                    self.calibration.calculate_units_for_volume(volume).loc[
                        self.calibration.calculate_units_for_volume(volume)[
                            "Well"] == well][
                        "Units Needed"].values[0]

                # sample the coefficient
                unit_value = (unit_value *
                              sdl_config.liquid_sampler_sample_coefficient)
                unit_values.append(unit_value)
            logger.info(
                f"Calibrating sampling to {unit_values} units. {volumes} "
                f"uL (with coefficient "
                f"{sdl_config.liquid_sampler_sample_coefficient})")
            kwargs["volumes"] = unit_values
            return func(self, **kwargs, )

        return wrapper

    def __init__(self, device_id: Optional[ObjectId] = None,
                 device_name: str = ""):
        if device_id is None:
            device_id = ObjectId()
        super().__init__(device_id, device_name)
        self.device_id = device_id
        self.ip = sdl_config.liquid_sampler_ip
        self.port = sdl_config.liquid_sampler_port
        self.connection_string = f"http://{self.ip}:{self.port}"

        self.dao = LiquidSamplerDAO(device_id=self.device_id)
        self.device_id = self.dao.create_entry(device_name, self.status)

        self.calibration = LiquidSamplerCalibration(
            sdl_config.liquid_sampler_calibration_file,
            sdl_config.liquid_sampler_calibration_cache_file)

        self._update_status(DeviceStatus.IDLE)

    def run(self,
            program: str,
            task_id: Optional[ObjectId] = None,
            experiment_id: Optional[ObjectId] = None,
            *args,
            **kwargs) -> None:
        """
        This method runs a program on the plate reader.
        """
        # This will call an attribute of the class.

        self._update_status(DeviceStatus.RUNNING)
        self.__getattribute__(program)(*args, **kwargs)
        self._update_status(DeviceStatus.IDLE)

    def plate_out(self, *args, **kwargs):
        logger.info("Plate out method called")
        response = requests.get(f"{self.connection_string}/plate_out", )
        response.raise_for_status()

    def plate_in(self, *args, **kwargs):
        logger.info("Plate in method called")
        response = requests.get(f"{self.connection_string}/plate_in", )
        response.raise_for_status()

    @_calibrate_volume_decorator
    def pump_by_motor_number(self, motor_number: int, volume: float, *args,
                             **kwargs):
        logger.info(f"Pump_by_motor_number method called, activating "
                    f"{motor_number},"
                    f" {volume} unit.")
        response = requests.post(f"{self.connection_string}/pump",
                      json={
                          "motor_number": str(motor_number),
                          "volume": str(volume)
                      })
        response.raise_for_status()
        logger.info("Pump_by_motor_number method done.")

    @_calibrate_volume_decorator
    def aspirate_by_motor_number(self, motor_number: int, volume: float, *args,
                                 **kwargs):
        logger.info(f"aspirate_by_motor_number method called, activating "
                    f"{motor_number}, {volume} unit.")
        result = requests.post(f"{self.connection_string}/aspirate", json={
            "motor_number": str(motor_number),
            "volume": str(volume)
        })
        result.raise_for_status()
        logger.info("aspirate_by_motor_number method done.")

    def aspirate_by_well(self, well: str, volume: float, *args, **kwargs):
        logger.info(f"aspirate_by_well method called, activating {well}")
        motor_number = _map_well_to_motor_number(well)
        self.aspirate_by_motor_number(motor_number=motor_number,
                                      volume=volume)
        logger.info("aspirate_by_well method done.")

    def pump_by_well(self, well: str, volume: float, *args, **kwargs):
        logger.info(f"Pump_by_well method called, activating {well}")
        motor_number = _map_well_to_motor_number(well)
        self.pump_by_motor_number(motor_number=motor_number,
                                  volume=volume)
        logger.info("Pump_by_well method done.")

    @_calibrate_volumes_decorator
    def pump_batch_by_motor_number(self, motor_numbers: List[float],
                                   volumes: List[int],
                                   *args, **kwargs):
        for motor_number, amount in zip(motor_numbers, volumes):
            logger.info(f"Pump_batch_by_motor_number method called, activating"
                        f" {motor_number}, {amount} unit")
        response = requests.post(f"{self.connection_string}/batch_pump", json={
            "motor_numbers": [str(motor_number) for motor_number in
                              motor_numbers],
            "volumes": [str(amount) for amount in volumes]
        })
        response.raise_for_status()
        logger.info("Pump_batch_by_motor_number method done.")

    @_calibrate_volumes_decorator
    def aspirate_batch_by_motor_number(self,
                                       motor_numbers: List[float],
                                       volumes: List[int],
                                       *args, **kwargs):
        for motor_number, amount in zip(motor_numbers, volumes):
            logger.info(f"aspirate_batch_by_motor_number method called, "
                        f"activating {motor_number}, {amount} unit")
        response = requests.post(f"{self.connection_string}/batch_aspirate", json={
            "motor_numbers": [str(motor_number) for motor_number in
                              motor_numbers],
            "volumes": [str(amount) for amount in volumes]
        })
        response.raise_for_status()
        logger.info("aspirate_batch_by_motor_number method done.")

    def pump_batch_by_well(self, wells: List[str], volumes: List[int],
                           *args, **kwargs):
        """
        This method pumps a batch by well.

        Args:
            wells (List[str]): the wells to pump
            volumes (List[int]): the volumes to pump
        """

        logger.info("Pump_batch_by_well method called")
        motor_numbers = [_map_well_to_motor_number(well) for well in wells]
        self.pump_batch_by_motor_number(motor_numbers=motor_numbers,
                                        volumes=volumes)
        logger.info("Pump_batch_by_well method done.")
        
    def aspirate_batch_by_well(self, wells: List[str], volumes: List[int],
                                 *args, **kwargs):
        """
        This method aspirates a batch by well.

        Args:
            wells (List[str]): the wells to aspirate
            volumes (List[int]): the volumes to aspirate
        """

        logger.info("Aspirate_batch_by_well method called")
        motor_numbers = [_map_well_to_motor_number(well) for well in wells]
        self.aspirate_batch_by_motor_number(motor_numbers=motor_numbers,
                                            volumes=volumes)
        logger.info("Aspirate_batch_by_well method done.")

    def sample_reagents(self, reagent_list: List[str],
                        volumes: List[int],
                        *args, **kwargs):
        """
        This method samples reagents.

        Args:
        """
        logger.info("Sampling reagents")
        motor_numbers = _map_reagents_to_motor_numbers(reagent_list)
        self.pump_batch_by_motor_number(motor_numbers, volumes)
        logger.info("Sampling reagents done.")

    def _update_status(self, status: DeviceStatus) -> None:
        self.status = status
        self.dao.update_status(status)

    def connect(self) -> None:
        try:
            if self._is_connected():
                self._update_status(DeviceStatus.IDLE)
            else:
                self._update_status(DeviceStatus.ERROR)
        except requests.RequestException:
            self._update_status(DeviceStatus.ERROR)

    def disconnect(self) -> bool:
        return True

    def is_running(self) -> bool:
        return self.status == DeviceStatus.RUNNING

    def is_idle(self) -> bool:
        return self.status == DeviceStatus.IDLE

    def _is_connected(self) -> bool:
        """
        This method checks if the liquid sampler is connected through
        a GET request.
        """
        try:
            # try to connect to the device
            response = requests.get(f"{self.connection_string}/")
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.RequestException:
            return False
