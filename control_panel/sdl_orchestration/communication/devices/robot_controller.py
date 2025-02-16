import socket
import time
from typing import Dict, Optional

from bson import ObjectId

from sdl_orchestration import logger, sdl_config
from sdl_orchestration.communication.callbacks.base_callback import \
    CallbackState
from sdl_orchestration.communication.device_controller import \
    BaseDeviceController, DeviceStatus
from sdl_orchestration.database.daos import RobotDAO

from sdl_orchestration.utils import parse_human_readable_time

BUFF_SIZE = 1024


class RobotController(BaseDeviceController):

    def __init__(self,
                 device_id: Optional[ObjectId] = None,
                 device_name: str = ""):
        if device_id is None:
            device_id = ObjectId()

        super().__init__(device_id, device_name)
        self.device_id = device_id
        self.robot_ip = sdl_config.robot_ip
        self.robot_port = sdl_config.robot_port
        self.socket = None
        self.dao = RobotDAO(device_id=self.device_id)
        self.device_id = self.dao.create_entry(device_name, self.status)

    def stop(self):
        """
        This method stops the robot.
        """
        self._send_command("stop")
        response = self._receive_response()
        if "Stopping program" in response:
            logger.info("Program stopped...")
        else:
            logger.error(f"Program stopping failed: {response}")
            self._update_status(DeviceStatus.ERROR)
            raise Exception(f"Program stopping failed: {response}")

    def run(self, program: str,
            task_id: Optional[ObjectId] = None,
            experiment_id: Optional[ObjectId] = None,
            callback=None) -> None:
        """
        This method runs a program on the robot.

        Args:
            program (str): The program to be run on the robot. This program
        is a URScript program (.urp).
            task_id (ObjectId): The task id
        associated with the program.
            experiment_id (ObjectId): The experiment
        id associated with the program.
        """
        if self.is_running():
            logger.error("Robot is already running a program")
            self._update_status(DeviceStatus.ERROR)
            raise Exception("Robot is already running a program")
        elif self.is_idle():
            self.run_program(program, task_id, experiment_id, callback)
        else:
            logger.error("Robot is in an invalid state")

    def connect(self) -> None:
        """
        Connect to the UR5e using the dashboard server. This set a socket connection to the robot.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)
        self.socket.connect((self.robot_ip, self.robot_port))
        response = self._receive_response()
        if "Connected" in response:
            self._update_status(DeviceStatus.IDLE)
            logger.info(f"Robot({self.device_id}) is connected")
        else:
            logger.error(
                f"Robot({self.device_id}) connection failed: {response}")

    #     TODO: power on the robot first
    #     TODO: activate the gripper

    def _send_command(self, command: str) -> None:
        """
        This method sends a command to the robot.

        Args:
            command (str): The command to be sent to the robot controller.
        """
        if not self._is_connected():
            self._update_status(DeviceStatus.ERROR)
            raise Exception("Robot controller is not connected")
        logger.debug(f"sending:{command}")
        if not command.endswith("\n"):
            command = command + "\n"
        to_send = bytes(command, encoding="ASCII")
        self.socket.sendall(to_send)

    def _receive_response(self) -> str:
        """
        This method receives a response from the robot.
        """
        if not self._is_connected():
            self._update_status(DeviceStatus.ERROR)
            raise Exception("Robot controller is not connected")
        reply = self.socket.recv(BUFF_SIZE)
        reply_ascii = reply.decode(encoding="ascii")
        logger.debug(f"receiving:{reply_ascii}")

        return reply_ascii

    def disconnect(self) -> None:
        """
        This function closes the socket connection to the robot.
        """
        if self._is_connected():
            self._send_command("quit")
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
            self.socket = None
            self._update_status(DeviceStatus.DISCONNECTED)
            logger.info("Robot controller is disconnected")
        else:
            logger.error("Disconnection failed: robot controller is not "
                         "connected")

    def _is_remote_running(self) -> bool:
        """
        Check whether the robot is running a program.
        """
        self._send_command("programState")
        response = self._receive_response()
        if "PLAYING" in response or "PAUSED" in response:
            return True
        else:
            return False

    def is_running(self) -> bool:
        """Check whether this device is running."""
        if self._is_connected() is False:
            logger.info("Robot controller is not connected")
            return False
        return self._is_remote_running()

    def is_idle(self) -> bool:
        """Check whether this device is idle."""
        if self._is_connected() is False:
            return False
        if self.status == DeviceStatus.IDLE:
            return True
        else:
            return False

    def run_program(self,
                    program: str,
                    task_id: Optional[ObjectId] = None,
                    experiment_id: Optional[ObjectId] = None,
                    callback: Optional[callable] = None
                    ) -> None:
        """
        This method loads and runs a program on the robot.

        Args:
            program (str): The program to be run on the robot.
            task_id (ObjectId): The task id associated with the program.
            experiment_id (ObjectId): The experiment id associated with
                                    the program.
            callback (callable): The callback function to be called during running.
        """
        self.dao.log_step(program, task_id, experiment_id)

        self._load_program(program)
        logger.info(f"Program {program} loaded to robot controller")
        time.sleep(0.5)
        self._start_program()
        self._wait_for_program_to_finish(callback)
        logger.info(
            f"Program {program} finished running, with task id {task_id} "
            f"and experiment id {experiment_id}")

    def _load_program(self, program: str) -> None:
        """
        Load a program to the robot, and validate the output.

        Args:
            program (str): The program to be loaded to the robot.
        """
        self._send_command(f"load {program}.urp")
        response = self._receive_response()
        if "Loading program" in response:
            logger.info(f"Program loaded successfully {program}")
        else:
            logger.error(f"Program loading failed: {response}")
            self._update_status(DeviceStatus.ERROR)
            raise Exception(f"Program loading failed: {response}")

    def _start_program(self, ) -> None:
        """
        This method starts a program on the robot.
        """
        self._send_command(f"play")
        response = self._receive_response()
        if "Starting program" in response:
            logger.info(f"Program started successfully.")
            self._update_status(DeviceStatus.RUNNING)
        else:
            logger.error(f"Program starting failed: {response}")
            self._update_status(DeviceStatus.ERROR)
            raise Exception(f"Program starting failed: {response}")

    def _wait_for_program_to_finish(self,
                                    callback: Optional[callable] = None, ):
        """
        This method waits for the program to finish running.
        """
        current_state = None

        var_to_query_list = callback.variables if callback is not None else []

        time.sleep(0.5)

        start_time = time.time()

        time_limit = sdl_config.robot_time_limit

        while self.is_running():
            if time.time() - start_time > time_limit:
                logger.error("Program running time exceeded the limit; possibly"
                             " due to an protective stop.")
                self.stop()
                self._update_status(DeviceStatus.ERROR)
                raise Exception(f"Robot running time exceeded the limit"
                                f" : {parse_human_readable_time(time_limit)} "
                                f"minutes, possibly due to an protective stop.")
            if callback is not None:
                variable_dict = self.query_variable(var_to_query_list)
                instruction = callback(variable_dict)
                if instruction == current_state:
                    continue
                if instruction == CallbackState.finish:
                    continue
                elif instruction == CallbackState.pause:
                    current_state = CallbackState.pause
                    self.pause()
                elif instruction == CallbackState.play:
                    current_state = CallbackState.play
                    self.play()
                else:
                    logger.error(f"Invalid instruction: {instruction}")
                    raise Exception(f"Invalid instruction: {instruction}")
            time.sleep(0.5)

        logger.info(f"Program finished running, time elapsed: "
                    f"{time.time() - start_time}")


    def query_variable(self,
                       var_to_query_dict: Dict[str, str]) -> Dict[str, str]:
        """
        This method queries a variable from the robot.

        Args:
            var_to_query_dict (Dict[str, str]): The variable to be queried.

        Returns:
            str: The value of the variable.
        """
        res_dict = {}
        for var in var_to_query_dict:
            self._send_command(f"getVariable {var}")
            response = self._receive_response()
            if response:
                # logger.debug(f"Variable {var} queried successfully: {response}")
                res_dict[var] = response
            else:
                logger.error(f"Variable {var} querying failed: {response}")
                self._update_status(DeviceStatus.ERROR)
                raise Exception(f"Variable {var} querying failed: {response}")
        return res_dict

    def pause(self) -> None:
        """
        This method pauses the robot.
        """
        self._send_command("pause")
        response = self._receive_response()
        if "Pausing program" in response:
            logger.info("Program paused...")
        else:
            logger.error(f"Program pausing failed: {response}")
            self._update_status(DeviceStatus.ERROR)
            raise Exception(f"Program pausing failed: {response}")

    def play(self) -> None:
        self._send_command("play")
        response = self._receive_response()
        if "Playing program" in response or "Starting" in response:
            logger.info("Program playing...")
        else:
            logger.error(f"Program playing failed: {response}")
            self._update_status(DeviceStatus.ERROR)
            raise Exception(f"Program playing failed: {response}")

    def _is_connected(self) -> bool:
        """
        Check whether the robot controller is connected.
        """
        if self.status == DeviceStatus.DISCONNECTED and self.socket is None:
            return False
        # elif self.status != DeviceStatus.DISCONNECTED and self.socket is
        # None: raise Exception("Robot controller is in an invalid state")
        # elif self.status == DeviceStatus.DISCONNECTED and self.socket is
        # not None: raise Exception("Robot controller is in an invalid state")
        else:
            return True

    def _update_status(self, status: DeviceStatus) -> None:
        """
        Update the status of the robot controller.

        Args:
            status (DeviceStatus): The status of the robot controller.
        """
        self.status = status
        self.dao.update_status(status)
