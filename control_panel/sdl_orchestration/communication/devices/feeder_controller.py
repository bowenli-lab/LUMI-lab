import time

import requests
from typing import Dict, List, Optional

from bson import ObjectId

from sdl_orchestration.communication.callbacks import BaseCallback
from sdl_orchestration.communication.callbacks.base_callback import \
    CallbackState
from sdl_orchestration.communication.device_controller import (
    BaseDeviceController,
    DeviceStatus,
)
from sdl_orchestration.database.daos import FeederDAO
from sdl_orchestration import logger, sdl_config


class Feedable:
    """
    Abstract class for feedable devices.
    """

    def feed_plate(self, cargo_num: int):
        """
        This method feeds the plate.
        """
        pass

    def feed_plate_for_callback(self, cargo_num: int) -> BaseCallback:
        """
        This method feeds the plate. It returns a callback that will be used
        to check the status of the feeder and determine the next action.
        """
        pass

    def is_running(self) -> bool:
        """
        This method checks if the feeder is running.
        """
        pass


class FeederLoadCallback(BaseCallback):
    current_feeder: Feedable
    """
    A callback class to manage the loading process of a feeder system.
    """

    def __init__(self,
                 variables: List[str],
                 item_num: int,
                 current_feeder: Feedable,
                 ):
        self.variables = variables
        self.item_num = item_num

        self.current_item_num_count = 0
        self.current_feeder = current_feeder
        self.plate_counts = {loop: 0 for loop in range(1, item_num + 1)}

    def __call__(self,
                 variables_query_dict: Dict[str, str],
                 ) -> str:
        """
        This method determines the action based on the "Loop_1" variable in
        the variables_query_dict.


        Args:
            variables_query_dict (Dict[str, str]): The dictionary containing
             query variables.

        Returns:
            str: The action to be taken ("finish", "pause", or "play").
                - "finish": Indicates the end of the process if the current loop exceeds
                   the total number of items.
                - "pause": Pauses the robot if it's the first time entering the current loop.
                - "play": Continues the process and calls the feeder to feed the plate
                   if no plate is being fed in the current loop.
                - "ERROR": Returns this if "Loop_1" is not found in the variables_query_dict
                   or if there's an inconsistency in the loop handling.
        """
        if "Loop_1" in variables_query_dict:
            time.sleep(0.1)
            current_loop = int(variables_query_dict["Loop_1"]) + 1

            if (current_loop in self.plate_counts
                    and self.plate_counts[current_loop] == 1):
                return CallbackState.play

            elif current_loop > self.item_num:
                return CallbackState.finish

            elif current_loop > self.current_item_num_count:
                # Pause the robot as this is the first time entering the loop
                self.current_item_num_count += 1
                return CallbackState.pause

            elif (current_loop == self.current_item_num_count
                  and self.plate_counts[current_loop] == 0):
                # call the feeder to feed the plate if no plate is being fed
                # in this loop
                self.current_feeder.feed_plate(1)
                self.plate_counts[current_loop] += 1
                return CallbackState.play

        return "ERROR"


class FeederController(BaseDeviceController,
                       Feedable):
    def __init__(
            self,
            device_id: Optional[ObjectId] = None,
            device_code: int = 0,
            device_name: str = "",
    ):
        """
        This is the constructor for the FeederController class.

        Args: device_id (Optional[ObjectId], optional): The device ID.
        Defaults to None. device_code (int, optional): The device code,
        indicate which device of this type it is. Defaults to 0. device_name
        (str, optional): The device name. Defaults to "".
        """
        self.client = None
        if device_id is None:
            device_id = ObjectId()

        super().__init__(device_id, device_name)
        self.device_id = device_id
        self.device_code = device_code
        if device_code not in sdl_config.feeder_config:
            raise ValueError(
                f"Device code {device_code} not found in config."
                f" Please register the device in the config toml file."
            )
        this_device_config = sdl_config.feeder_config[device_code]
        self.ip = this_device_config["host"]
        self.port = this_device_config["port"]
        self.connection_string = f"http://{self.ip}:{self.port}"
        self.dao = FeederDAO(device_id=self.device_id, device_code=device_code)
        self.status = DeviceStatus.IDLE
        self.auto_feed = True  # Deprecated
        self.device_id = self.dao.create_entry(device_name, self.status)

    def run(
            self,
            program: str,
            task_id: Optional[ObjectId] = None,
            experiment_id: Optional[ObjectId] = None,
    ) -> None:
        """
        This method runs a program on the feeder.
        """
        self.__getattribute__(program)()

    def feed_plate(self, cargo_num: int):
        logger.info("Feeding plate method called")
        self._update_status(DeviceStatus.RUNNING)
        try:
            response = requests.get(
                f"{self.connection_string}/move_up_cargo/{cargo_num}")
            response.raise_for_status()
            logger.info(f"Feed plate successful: {response.json()}")
        except requests.RequestException as e:
            logger.error(f"Error during feed plate: {e}")
            raise e
        finally:
            self._update_status(DeviceStatus.IDLE)

    def feed_plate_for_callback(self, cargo_num: int) -> BaseCallback:
        """
        This method feeds the plate. It returns a callback that will be used
        to check the status of the feeder and determine the next action.

        Args:
            cargo_num (int): The number of cargos to move the plate by.
        """

        return FeederLoadCallback(["Loop_1"], cargo_num, self)

    def feed_plate_by_type(self, cargo_type: str, cargo_num: int):
        """
        This method feeds the plate by the given number of cargos.

        Args:
            cargo_type (str): The type of cargo to move the plate by.
            cargo_num (int): The number of cargos to move the plate by.
        """
        logger.info("Feeding plate by type method called")
        self._update_status(DeviceStatus.RUNNING)
        try:
            response = requests.get(
                f"{self.connection_string}/move_up_by_type/{cargo_type}/{cargo_num}"
            )
            response.raise_for_status()
            logger.info(
                f"Feed  {cargo_type} by {cargo_num} successful: "
                f"{response.json()}")
        except requests.RequestException as e:
            logger.error(f"Error during feed plate by type: {e}")
        finally:
            self._update_status(DeviceStatus.IDLE)

    def feed_plate_down_by_type(self, cargo_type: str, cargo_num: int):
        """
        This method moves the plate down by the given number of cargos.

        Args:
            cargo_type (str): The type of cargo to move the plate by.
            cargo_num (int): The number of cargos to move the plate by.
        """
        logger.info("Feeding plate down by type method called")
        self._update_status(DeviceStatus.RUNNING)
        try:
            response = requests.get(
                f"{self.connection_string}/move_down_by_type/{cargo_type}/{cargo_num}"
            )
            response.raise_for_status()
            logger.info(
                f"Feed down {cargo_type} by {cargo_num} successful: "
                f"{response.json()}")
        except requests.RequestException as e:
            logger.error(f"Error during feed plate down by type: {e}")
        finally:
            self._update_status(DeviceStatus.IDLE)

    def feed_up(self, steps: int = 1):
        """
        This method feeds the plate up by the given number of steps.
        """
        logger.info("Feeding up method called")
        self._update_status(DeviceStatus.RUNNING)
        try:
            response = requests.get(f"{self.connection_string}/move_up/{steps}")
            response.raise_for_status()
            logger.info(f"Feed up successful: {response.json()}")
        except requests.RequestException as e:
            logger.error(f"Error during feed up: {e}")
        finally:
            self._update_status(DeviceStatus.IDLE)

    def feed_down(self, steps: int = 1):
        """
        This method moves the plate down by the given number of steps.
        """
        logger.info("Feeding down method called")
        self._update_status(DeviceStatus.RUNNING)
        try:
            response = requests.get(
                f"{self.connection_string}/move_down/{steps}")
            response.raise_for_status()
            logger.info(
                f"{self.device_code} | Feed down successful: {response.json()}")
        except requests.RequestException as e:
            logger.error(f"{self.device_code} | Error during feed down: {e}")
        finally:
            self._update_status(DeviceStatus.IDLE)

    def feed_bottom(self):
        """
        This method moves the plate to the bottom.
        """
        logger.info("Feeding to bottom method called")
        self._update_status(DeviceStatus.RUNNING)
        try:
            response = requests.get(f"{self.connection_string}/move_bottom")
            response.raise_for_status()
            logger.info(
                f"{self.device_code} | Feed to bottom successful: {response.json()}")
        except requests.RequestException as e:
            logger.error(
                f"{self.device_code} | Error during feed to bottom: {e}")
        finally:
            self._update_status(DeviceStatus.IDLE)

    def enable_auto_feed(self) -> None:
        """
        This method enables auto feed.
        """
        self._set_auto_feed(True)

    def disable_auto_feed(self) -> None:
        """
        This method disables auto feed.
        """
        self._set_auto_feed(False)

    def _set_auto_feed(self, auto_feed: bool) -> None:
        """
        This method sets the auto feed status of the feeder.
        """
        self.auto_feed = auto_feed
        try:
            if auto_feed:
                response = requests.get(
                    f"{self.connection_string}/enable_auto_feed/{auto_feed}"
                )
            else:
                response = requests.get(
                    f"{self.connection_string}/disable_auto_feed/{auto_feed}"
                )
            logger.info(f"{self.device_code} | Auto feed set to {auto_feed}, "
                        f"{response.json()}")
        except requests.RequestException as e:
            logger.error(f"{self.device_code} | Error setting auto feed: {e}")

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
        response = requests.get(f"{self.connection_string}/is_running")
        try:
            res = response.json()
            return res["is_running"]
        except Exception:
            raise Exception("Error checking if the feeder is running.")

    def is_idle(self) -> bool:
        return self.status == DeviceStatus.IDLE

    def _update_status(self, status: DeviceStatus) -> None:
        self.status = status
        self.dao.update_status(status)

    def _is_connected(self) -> bool:
        """
        This method checks if the feeder is connected through a GET request.
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
