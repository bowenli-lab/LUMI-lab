import requests
from sdl_orchestration.communication.device_controller import \
    BaseDeviceController, DeviceStatus
import time
from typing import Any, Dict, Optional, Tuple
from bson import ObjectId
from sdl_orchestration import logger, sdl_config
from sdl_orchestration.database.daos import PlateReaderDAO
from sdl_orchestration.notification.client import notification_client


class PlateReaderController(BaseDeviceController):

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def is_running(self) -> bool:
        return self.status == DeviceStatus.RUNNING

    def is_idle(self) -> bool:
        return self.status == DeviceStatus.IDLE

    def __init__(self, device_id: Optional[ObjectId] = None,
                 device_name: str = ""):
        if device_id is None:
            device_id = ObjectId()
        super().__init__(device_id, device_name)
        self.device_id = device_id
        self.ip = sdl_config.plate_reader_ip
        self.port = sdl_config.plate_reader_port
        self.connection_string = f"http://{self.ip}:{self.port}"
        self.base_url = f"{self.connection_string}"

        self.status = DeviceStatus.IDLE  # change this later

        self.plate_reader_dao = PlateReaderDAO(device_id=self.device_id)
        self.device_id = self.plate_reader_dao.create_entry(object_name=str(self.__class__.__name__),
                                           status=self.status)

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

        # self._update_status(DeviceStatus.RUNNING)
        self.__getattribute__(program)(*args, **kwargs)
        # self._update_status(DeviceStatus.IDLE)

    def _before_run(self):
        self._update_status(DeviceStatus.RUNNING)

    def _update_status(self, status: DeviceStatus):
        self.plate_reader_dao.update_status(status)
        self.status = status

    def _after_run(self):
        self._update_status(DeviceStatus.IDLE)

    def _send_notification(self,
                           task_name: str,
                           progress: str,
                           experiment_id: Optional[str] = None,
                           elapsed_time: str = None) -> None:
        """
        This method sends a notification to the Slack channel.
        """
        notification_client.send_notification(
            message_type="progress",
            task_name=task_name,
            progress=progress,
            experiment_id=experiment_id,
            elapsed_time=elapsed_time)

    def carrier_in(self):
        self._before_run()
        url = f"{self.base_url}/carrier-in"
        try:
            response = requests.post(url, timeout=600)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error reading plate: {e}, retrying...")
            time.sleep(300)
            response = requests.post(url, timeout=600)
            response.raise_for_status()
            self._after_run()
            return response.json()
        self._after_run()

    def carrier_out(self):
        self._before_run()
        url = f"{self.base_url}/carrier-out"
        try:
            response = requests.post(url, timeout=600)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error reading plate: {e}, retrying...")
            time.sleep(300)
            response = requests.post(url, timeout=600)
            response.raise_for_status()
            self._after_run()
            return response.json()
        self._after_run()

    def read_plate(self, protocol_path, experiment_path, csv_path):
        self._before_run()
        url = f"{self.base_url}/read-plate"
        url = url + f"?protocol_path={protocol_path}&experiment_path={experiment_path}&csv_path={csv_path}"
        try:
            response = requests.post(url, timeout=600)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error reading plate: {e}, retrying...")
            time.sleep(300)
            response = requests.post(url, timeout=600)
            response.raise_for_status()
            self._after_run()
            return response.json()
        self._after_run()
        return response.json()

    def parse_plate(self, csv_path: str) -> tuple[Any, Any]:
        self._before_run()
        url = f"{self.base_url}/parse-file"
        url = url + f"?csv_path={csv_path}"
        try:
            response = requests.post(url, timeout=600)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Error reading plate: {e}, retrying...")
            time.sleep(1)
            response = requests.post(url, timeout=600)
            response.raise_for_status()
            self._after_run()
            return response.json()
        if "error" in response.json():
            # try again
            logger.error("Error in parse plate")
            time.sleep(60)
            response = requests.post(url, timeout=300)
            response.raise_for_status()
        self._after_run()
        reading_info = response.json()["reading_info"]
        results = response.json()["results"]

        self._send_notification(
                        task_name="Reading Results",
                        progress=reading_info)

        return reading_info, results
