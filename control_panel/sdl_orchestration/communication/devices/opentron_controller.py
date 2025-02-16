import json
import os
import select
import time
from typing import List, Optional

import requests
from bson import ObjectId

from sdl_orchestration.communication.device_controller import (
    BaseDeviceController,
    DeviceStatus,
)

from sdl_orchestration.database.daos import OpentronDAO

from sdl_orchestration import logger, sdl_config

from sdl_orchestration.utils import parse_human_readable_time

from sdl_orchestration.notification.client import notification_client


class OpentronException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        notification_client.send_notification(
            message_type="error",
            task_name="Error",
            error_message=message,
            mention_all=True,
        )


class OpentronController(BaseDeviceController):
    def __init__(
        self,
        device_id: Optional[ObjectId] = None,
        device_code: int = 0,
        device_name: str = "",
    ):
        self.client = None
        if device_id is None:
            device_id = ObjectId()

        super().__init__(device_id, device_name)
        self.device_id = device_id
        self.device_code = device_code
        self.dao = OpentronDAO(device_id=self.device_id, device_code=device_code)
        self.status = DeviceStatus.IDLE
        self.device_id = self.dao.create_entry(device_name, self.status)
        self.config = sdl_config.opentron_config[device_code]
        self.robot_ip = self.config["host"]
        self.program_path = self.config["program_path"]
        self.api_version = "3"
        self.headers = {
            "Opentrons-Version": self.api_version,
            "Content-Type": "application/json",
        }
        self.port = self.config["port"]
        if "parsed_synthesis_scripts_path" in self.config:
            self.parsed_synthesis_scripts_path = (self.config)[
                "parsed_synthesis_scripts_path"
            ]
            logger.info(
                f"Using parsed synthesis scripts path: "
                f"{self.parsed_synthesis_scripts_path}"
            )
        self.current_run_id = None

    def submit_run(
        self,
        program: str,
        task_id: Optional[ObjectId] = None,
        experiment_id: Optional[ObjectId] = None,
        *args,
        **kwargs,
    ) -> str:
        """
        This method submits a run to the Opentron. It runs the program
        asynchronously and returns the run ID.
        """
        if not self._is_connected():
            logger.error("Opentron is not connected")
            raise OpentronException("Opentron is not connected")
        self.dao.log_step(program, task_id, experiment_id)
        self._update_status(DeviceStatus.RUNNING)
        logger.info(
            f"Opentron {self.device_code} is running program {program}"
            f"asynchronously."
        )
        self.send_notification(program, "Starting async running", str(experiment_id))

        protocol_id = self._upload_protocol(program)
        run_id = self._create_run(protocol_id)
        self.current_run_id = str(run_id)
        self._start_run(run_id)

        return str(run_id)

    def query_run(self, run_id: str) -> str:
        """
        This method queries the status of a run on the Opentron.

        Args:
            run_id (str): The run ID.
        """
        url = f"http://{self.robot_ip}:{self.port}/runs/{run_id}"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            logger.error(f"Failed to query run: {response.text}")
            raise OpentronException("Failed to query run")
        run_status = response.json()["data"]["status"]
        if run_status == "succeeded":
            logger.info(f"Run {run_id} succeeded on controller.")
            self._update_status(DeviceStatus.IDLE)
            self.current_run_id = None

        return run_status

    def _wait_for_run(self, run_id: str) -> None:
        #  Wait for the run to complete
        time_elapsed = 0

        while True:
            time_start = time.time()
            run_status = self.query_run(run_id)
            if run_status == "succeeded":
                logger.info(f"Run {run_id} succeeded.")
                break
            elif run_status == "running" or run_status == "finishing":
                pass
            else:
                logger.error(f"Run {run_id} errored, status: {run_status}")
                raise OpentronException("Run errored")
            time.sleep(10)
            time_end = time.time()
            time_elapsed += time_end - time_start
            time_elapsed_str = parse_human_readable_time(int(time_elapsed))
            logger.info(
                f"Time elapsed: {time_elapsed_str} | " f"Run status: {run_status}"
            )

    def run(
        self,
        program: str,
        task_id: Optional[ObjectId] = None,
        experiment_id: Optional[ObjectId] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        This method runs the program on the Opentron synchronously. It uploads
        the protocol and labware files to the Opentron, creates a run, and
        starts the run. It waits for the run to complete before returning.

        Args:
            program (str): The program file name.
            task_id (ObjectId, optional): The task ID. Defaults to None.
            experiment_id (ObjectId, optional): The experiment ID. Defaults to
                None.
        """

        if not self._is_connected():
            logger.error("Opentron is not connected")
            raise OpentronException("Opentron is not connected")

        self.dao.log_step(program, task_id, experiment_id)
        self._update_status(DeviceStatus.RUNNING)
        logger.info(f"Opentron {self.device_code} is running program {program}")
        self.send_notification(program, "Starting", str(experiment_id))
        time_start = time.time()
        protocol_id = self._upload_protocol(program)
        run_id = self._create_run(protocol_id)
        self.current_run_id = str(run_id)
        self._start_run(run_id)
        self._wait_for_run(run_id)
        time_end = time.time()
        elapsed_time = time_end - time_start
        elapsed_time_str = parse_human_readable_time(int(elapsed_time))
        self.send_notification(
            program, "Completed", str(experiment_id), elapsed_time_str
        )
        self.current_run_id = None

        self._update_status(DeviceStatus.IDLE)

    def send_notification(
        self,
        task_name: str,
        progress: str,
        experiment_id: str,
        elapsed_time: str = None,
    ) -> None:
        """
        This method sends a notification to the Slack channel.
        """
        notification_client.send_notification(
            message_type="progress",
            task_name=task_name,
            progress=progress,
            experiment_id=experiment_id,
            elapsed_time=elapsed_time,
        )

    def _upload_protocol(
        self,
        program: str,
    ) -> str:
        """
        This method uploads the protocol to the Opentron. The protocol is
        uploaded as a Python program (.py) file and the labware file is
        uploaded as a JSON file.

        Assumes that the program and labware files are in the program path(s)
        and labware path defined in the config file.

        Args:
            program (str): The program file name.
            labware_file (str, optional): The labware file name. Defaults to
        """
        url = f"http://{self.robot_ip}:{self.port}/protocols"
        program_full_path = self._parse_file_path(program)

        try:
            files = [("files", open(program_full_path, "rb"))]
            labware_path_list = self._get_all_labware_path_list()
            for labware_full_path in labware_path_list:
                files.append(("files", open(labware_full_path, "rb")))
            response = requests.post(
                url, headers={"Opentrons-Version": self.api_version}, files=files
            )
            # close the files
            for file in files:
                file[1].close()
        except IOError as e:
            logger.error(f"Error opening file: {e}")
            raise OpentronException("Failed to open protocol or labware file")

        if response.status_code != 201 and response.status_code != 200:
            logger.error(
                f"Failed to upload protocol: {response.status_code} \n"
                f"{response.text}"
            )
            raise OpentronException("Failed to upload protocol")

        logger.info(f"Uploaded protocol: {response.json()}")
        protocol_id = response.json()["data"]["id"]
        return protocol_id

    def _parse_file_path(self, program):
        """
        Parse the labware file path and program path. The program path is
        searched in the program path list defined in the config file.

        Args:
            program (str): The program file name.
        """
        if ".py" not in program:
            program = program + ".py"
        # find the program path that are in the program path list
        program_path = None
        # TODO: simlify the matching logic with task utils
        for path in self.config["program_path"]:
            if os.path.exists(f"{path}/{program}"):
                program_path = path
                break
        if program_path is None:
            logger.error(f"Program {program} not found in program path")
            raise OpentronException("Program not found")
        program_full_path = f"{program_path}/{program}"
        return program_full_path

    def _get_all_labware_path_list(self) -> List[str]:
        """
        Given the labware path directories in the config,
        match any .json under these directories.

        Return the list of all matched labware files.
        """
        labware_files = []
        for path in self.config["labware_path"]:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".json"):
                        logger.info(f"Found labware file: {file}")
                        labware_files.append(str(os.path.join(root, file)))

        return labware_files

    def _create_run(self, protocol_id: str, max_retries: int = 100) -> str:
        """
        This method creates a run from the uploaded protocol. It waits for the
        protocol analysis to complete before starting the run.

        Args:
            protocol_id (str): The protocol ID.

        """
        url = f"http://{self.robot_ip}:{self.port}/runs"
        payload = json.dumps({"data": {"protocolId": protocol_id}})
        response = requests.post(url, headers=self.headers, data=payload)
        if response.status_code != 201 and response.status_code != 200:
            logger.error(f"Failed to create run: {response.text}")
            raise OpentronException("Failed to create run")
        run_id = response.json()["data"]["id"]
        logger.info(f"Created run: {run_id}")

        # Wait for the protocol analysis to complete
        analysis_url = (
            f"http://{self.robot_ip}:{self.port}/protocols/" f"{protocol_id}/analyses"
        )
        for _ in range(max_retries):
            analysis_response = requests.get(analysis_url, headers=self.headers)
            if (
                analysis_response.status_code == 200
                or analysis_response.status_code == 201
            ):
                analysis_status = analysis_response.json()["data"][0]["status"]
                if analysis_status == "completed":
                    logger.info(f"{protocol_id} | Protocol analysis completed")
                    break
                elif analysis_status == "failed":
                    logger.error("Protocol analysis failed")
                    raise OpentronException("Protocol analysis failed")
            time.sleep(5)  # Wait for 5 seconds before retrying
        else:
            logger.error("Protocol analysis did not complete in time")
            raise OpentronException("Protocol analysis timeout")

        return run_id

    def _start_run(self, run_id: str) -> None:
        """
        This method starts the run on the Opentron. It waits for the run to
        complete before returning.

        Args:
            run_id (str): The run ID.
        """

        url = f"http://{self.robot_ip}:{self.port}/runs/{run_id}/actions"
        payload = json.dumps({"data": {"actionType": "play"}})
        response = requests.post(url, headers=self.headers, data=payload)
        if response.status_code != 200 and response.status_code != 201:
            logger.error(f"Failed to start run: {response.text}")
            raise OpentronException("Failed to start run")
        logger.info(f"Started run {run_id}")

    def _stop_run(self, run_id: str) -> None:
        """
        This method stops the run on the Opentron.
        """
        logger.info(f"Stopping run {run_id}")
        url = f"http://{self.robot_ip}:{self.port}/runs/{run_id}/actions"
        payload = json.dumps({"data": {"actionType": "stop"}})
        response = requests.post(url, headers=self.headers, data=payload)
        if response.status_code != 200 and response.status_code != 201:
            logger.error(f"Failed to stop run: {response.text}")
            raise OpentronException("Failed to stop run")

    def stop(self) -> None:
        """
        This method stops the Opentron run.
        """
        logger.info(f"Stopping Opentron {self.device_code}")
        if self.current_run_id is None:
            logger.error(f"Opentron {self.device_code}: No run to stop")
        self._stop_run(self.current_run_id)
        logger.info(f"Opentron {self.device_code}: Stopped run {self.current_run_id}")
        self._update_status(DeviceStatus.IDLE)

    def _check_results(self) -> None:
        raise NotImplementedError

    def _run_program(self, program: str) -> None:
        raise NotImplementedError

    def connect(self) -> None:
        """
        Connect to the Opentron using the SSH connection. The authentication
        is done using the SSH key defined in config file.
        """
        pass

    def disconnect(self) -> None:
        logger.info(f"Disconnected from Opentron {self.device_code}")

    def is_running(self) -> bool:
        return self.status == DeviceStatus.RUNNING

    def is_idle(self) -> bool:
        return self.status == DeviceStatus.IDLE

    def _update_status(self, status: DeviceStatus) -> None:
        self.status = status
        self.dao.update_status(status)

    def _is_connected(self) -> bool:
        # TODO add ping check
        return True
