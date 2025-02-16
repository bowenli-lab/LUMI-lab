import requests
from typing import Optional

from bson import ObjectId

from sdl_orchestration.communication.callbacks import BaseCallback
from sdl_orchestration.communication.device_controller import (
    BaseDeviceController,
    DeviceStatus,
)
from sdl_orchestration.database.daos import ClampDAO
from sdl_orchestration import logger, sdl_config


class ClampCallback(BaseCallback):
    current_clamp: BaseDeviceController

    def __init__(self,
                 variables: list[str],
                 current_clamp: BaseDeviceController):
        self.current_clamp = current_clamp
        self.variables = variables

    def __call__(self, variables_query_dict: dict[str, str]) -> str:
        """
        Determines the action based on the robot's status and performs clamp operations.

        This callback method performs the following checks and actions:
        1) Checks if the necessary keys ("isPlaced" and "readyToGet") are present in the variables_query_dict.
           - Returns "ERROR" if either key is missing.

        2) Checks if the robot has placed the plate in the clamp and the clamp is ready to be closed.
           - Closes the clamp and returns "pause" if the plate is placed (`isPlaced` is True) and the clamp is not yet closed.
           - Returns "play" if the plate is placed (`isPlaced` is True) and the clamp is already closed.

        3) Checks if the clamp is closed and the robot is ready to pick up the plate.
           - Releases the clamp and returns "pause" if the clamp is closed and the robot is ready to pick up the plate (`readyToGet` is True).
           - Returns "play" if the clamp is not closed and the robot is ready to pick up the plate (`readyToGet` is True).

        Args:
            variables_query_dict (dict[str, str]): The dictionary containing query variables.

        Returns:
            str: The action to be taken ("pause", "play", or "ERROR" if the required keys are not found).
        """
        # validate the variables
        if ("is_placed" not in variables_query_dict or "sealing_removed" not
                in variables_query_dict):
            return "ERROR"

        is_placed = str(variables_query_dict.get("is_placed", "")).strip() == "True"
        ready_to_get = str(variables_query_dict.get("sealing_removed", "")).strip() == "True"

        logger.debug(f"Is placed: {is_placed}, Ready to get: {ready_to_get},"
                     f" is_closed: {self.current_clamp.is_closed}")

        if is_placed and not self.current_clamp.is_closed:
            self.current_clamp.clamp()
            return "pause"
        elif is_placed and self.current_clamp.is_closed:
            return "play"
        elif self.current_clamp.is_closed and not ready_to_get:
            return "play"
        elif self.current_clamp.is_closed and ready_to_get:
            self.current_clamp.release()
            return "pause"
        elif not self.current_clamp.is_closed and ready_to_get:
            return "play"


class ClampController(BaseDeviceController):
    def __init__(
            self,
            device_id: Optional[ObjectId] = None,
            device_name: str = "",
    ):
        """
        This is the constructor for the Clamp class.

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
        this_device_config = sdl_config.clamp_configs
        self.ip = this_device_config["host"]
        self.port = this_device_config["port"]
        self.connection_string = f"http://{self.ip}:{self.port}"
        self.dao = ClampDAO(device_id=self.device_id)
        self.is_closed = False
        self.status = DeviceStatus.IDLE
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

    def clamp(self) -> None:
        """
        This method clamps the plate.
        """
        if self.is_closed:
            logger.info("Clamp is already closed.")
            return
        try:
            response = requests.get(f"{self.connection_string}/clamp")
            logger.info(f"Clamped: {response.json()}")
        except requests.RequestException as e:
            logger.error(f"Error clamping: {e}")
        self.is_closed = True
        self._update_status(DeviceStatus.RUNNING)

    def release(self) -> None:
        """
        This method releases the plate.
        """
        if not self.is_closed:
            logger.info("Clamp is already open.")
            return
        try:
            response = requests.get(f"{self.connection_string}/release")
            logger.info(f"Released: {response.json()}")
        except requests.RequestException as e:
            logger.error(f"Error releasing: {e}")
        self.is_closed = False
        self._update_status(DeviceStatus.IDLE)

    def auto_clamp(self) -> BaseCallback:
        """
        This method clamps the plate. It returns a callback that will be used
        to check the status of the clamp and determine the next action.
        """

        return ClampCallback(["is_placed", "sealing_removed"], self)


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
