import time
from typing import Optional

import requests
from bson import ObjectId

from sdl_orchestration.communication.device_controller import \
    BaseDeviceController, DeviceStatus

from sdl_orchestration.database.daos import IncubatorDAO

from sdl_orchestration import logger, sdl_config

SHARED_TIMER = 0


class IncubatorController(BaseDeviceController):

    def __init__(self,
                 device_id: Optional[ObjectId] = None,
                 device_name: str = "", ):
        self.client = None
        if device_id is None:
            device_id = ObjectId()

        super().__init__(device_id, device_name)
        self.device_id = device_id
        self.status = DeviceStatus.IDLE
        self.ip = sdl_config.incubator_ip
        self.port = sdl_config.incubator_port
        self.connection_string = f"http://{self.ip}:{self.port}"
        self.dao = IncubatorDAO(device_id=self.device_id)
        self.device_id = self.dao.create_entry(device_name, self.status)
        self.is_closed = True

    def run(self, wait_time: str,
            task_id: Optional[ObjectId] = None,
            experiment_id: Optional[ObjectId] = None) -> None:
        """
            This method times the incubator.
            """
        if not self._is_connected():
            logger.error("incubator is not connected")
            raise Exception("incubator is not connected")

        self.dao.log_step(wait_time, task_id, experiment_id)
        self._update_status(DeviceStatus.RUNNING)

        time.sleep(0.1)

        self._update_status(DeviceStatus.IDLE)

    def open_incubator(self):
        """
        This method opens the incubator door. It opens the door n times for
        pre-heating.

        Args:
            n: Number of times to open the door.
        """
        # for i in range(n):
        #     # open and close the incubator door n times for pre-heating
        #     self._open_incubator()
        #     self._close_incubator()
        # actually open the incubator door
        self._open_incubator()

    def _open_incubator(self):
        """
        This method opens the incubator door.
        """
        if not self.is_closed:
            logger.info("Incubator is already open.")
            raise Exception("Incubator is already open.")
        self._update_status(DeviceStatus.RUNNING)
        logger.info("Opening incubator door")
        try:
            response = requests.get(f"{self.connection_string}/open_incubator")
            if response.status_code == 200 or response.status_code == 201:
                self._update_status(DeviceStatus.IDLE)
            else:
                self._update_status(DeviceStatus.ERROR)
        except Exception as e:
            logger.error(f"Error opening incubator: {e}")
            self._update_status(DeviceStatus.ERROR)
        self._update_status(DeviceStatus.IDLE)
        self.is_closed = False

    def _close_incubator(self):
        """
        This method closes the incubator door.
        """
        if self.is_closed:
            logger.info("Incubator is already closed.")
            return
        self._update_status(DeviceStatus.RUNNING)
        logger.info("Closing incubator door")
        try:
            response = requests.get(f"{self.connection_string}/close_incubator")
            if response.status_code == 200 or response.status_code == 201:
                self._update_status(DeviceStatus.IDLE)
            else:
                self._update_status(DeviceStatus.ERROR)
        except Exception as e:
            logger.error(f"Error closing incubator: {e}")
            self._update_status(DeviceStatus.ERROR)
        self._update_status(DeviceStatus.IDLE)
        self.is_closed = True

    def close_incubator(self) -> None:
        logger.info("Incubator closing")
        self._close_incubator()

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def is_running(self) -> bool:
        return self.status == DeviceStatus.RUNNING

    def is_idle(self) -> bool:
        return self.status == DeviceStatus.IDLE

    def _update_status(self, status: DeviceStatus) -> None:
        self.status = status
        self.dao.update_status(status)

    def _is_connected(self) -> bool:
        return True

    def stop(self) -> None:
        logger.info("Incubator stopping")
        try:
            response = requests.get(f"{self.connection_string}/break")
            if response.status_code == 200 or response.status_code == 201:
                self._update_status(DeviceStatus.IDLE)
            else:
                self._update_status(DeviceStatus.ERROR)
        except Exception as e:
            logger.error(f"Error stopping incubator door: {e}")
            self._update_status(DeviceStatus.ERROR)
        self._update_status(DeviceStatus.IDLE)

