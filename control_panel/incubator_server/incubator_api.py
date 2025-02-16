import RPi.GPIO as GPIO
import logging
import os
import time

import board
from adafruit_motorkit import MotorKit

OPEN_DURATION = 16.6  # in seconds
CLOSE_DURATION = 18.0  # in seconds
DIRECTION_IN = -1
DIRECTION_OUT = 1

PLATE_MOVING_DURATION = 3


class IncubatorStatus:
    """
    The status of the incubator door.
    """
    CLOSED = 1
    OPEN = 0


class MotorStatus:
    """
    The status of the motor.
    """
    STOP = 0
    FORWARD = 1
    BACKWARD = -1


class IncubatorAPI:
    def __init__(
            self,
    ):
        self._is_shutting_down = False
        self.logger = self._config_logger()
        self.incubator_status = IncubatorStatus.CLOSED
        self.current_position = 0
        self.relay0 = 5
        self.relay1 = 6
        self.sensor_pin = 26
        self.motor = None
        self._init_motor()
        self.motor_status = MotorStatus.STOP
        self._config_relay()
        self._turn_on_sensor()
        self.logger.info("Incubator API initialized")
        self.shutdown_relay()

    def _config_logger(self):
        """
        This method configures the logger.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        file_handler = logging.FileHandler(
            f"./logs/incubator.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if __name__ == "__main__":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def _init_motor(self):
        """
        This method initializes the motor.
        """
        self.motor = MotorKit(i2c=board.I2C(),
                              address=0x64).motor3
        self.motor_status = MotorStatus.STOP

    def incubator_plate_out(self):
        """
        This method moves the incubator plate out.
        """
        if self.incubator_status == IncubatorStatus.OPEN:
            self.logger.info("Incubator is already open.")
            self.move_plate_out(PLATE_MOVING_DURATION)
        else:
            self.logger.info("Incubator is closed. Open the incubator first.")
            self.open_incubator()
        self.move_plate_out(PLATE_MOVING_DURATION)
        self.logger.info("Plate moved out")
        return "Plate moved out"

    def incubator_plate_in(self):
        """
        This method moves the incubator plate in.
        """
        if self.incubator_status == IncubatorStatus.OPEN:
            self.logger.info("Incubator is already open.")
            self.move_plate_in(PLATE_MOVING_DURATION)
        else:
            self.logger.info("Incubator is closed. Open the incubator first.")
            self.open_incubator()
        self.move_plate_in(PLATE_MOVING_DURATION)
        self.logger.info("Plate moved in")
        return "Plate moved in"

    def move_plate_out(self, duration: float):
        """
        This method moves the incubator plate out.

        Args:
            duration: The duration for which the motor should run.
        """
        self._move_motor(duration, MotorStatus.FORWARD)

    def move_plate_in(self, duration: float):
        """
        This method moves the incubator plate in.

        Args:
            duration: The duration for which the motor should run.
        """
        self._move_motor(duration, MotorStatus.BACKWARD)

    def _move_motor(self, duration: float, direction: int):
        """
        This method moves the motor in a given direction for a given duration.

        Args:
            duration: The duration for which the motor should run.
            direction: The direction in which the motor should run.
        """
        self.motor_status = direction
        self.motor.throttle = direction * 1
        end_time = time.time() + duration
        while time.time() < end_time:
            if self._is_shutting_down:
                self.logger.info("Operation stopped due to shutdown")
                break
            time.sleep(0.05)
        self.motor.throttle = 0
        self.motor_status = MotorStatus.STOP

    def _config_relay(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.relay0, GPIO.OUT)
        GPIO.setup(self.relay1, GPIO.OUT)

    def _turn_on_sensor(self):
        """
        This method turns on the postion sensor. It detects whether the incubator door is fully open.
        """
        GPIO.setup(self.sensor_pin, GPIO.OUT)
        GPIO.output(self.sensor_pin, GPIO.HIGH)

    def open_incubator(self):
        """
        This method opens the incubator door.
        """
        if self.incubator_status == IncubatorStatus.OPEN:
            self.logger.info("Incubator is already open.")
            return "Incubator is already open. Ignoring request."
        self.logger.info("Opening the incubator")
        self.door_forward(OPEN_DURATION)
        self.incubator_status = IncubatorStatus.OPEN
        self.shutdown_relay()
        return "Incubator opened."

    def close_incubator(self) -> str:
        """
        This method closes the incubator door.
        """
        if self.incubator_status == IncubatorStatus.CLOSED:
            self.logger.info("Incubator is already closed.")
            return "Incubator is already closed. Ignoring request."
        self.logger.info("Closing the incubator")
        self.door_backward(CLOSE_DURATION)
        self.incubator_status = IncubatorStatus.CLOSED
        self.shutdown_relay()
        return "Incubator closed."

    def door_forward(self, duration: float):
        """
        This method moves the incubator door forward.

        Args:
            duration: The duration for which the relay should be on.
        """
        GPIO.output(self.relay0, GPIO.HIGH)
        GPIO.output(self.relay1, GPIO.LOW)
        self._run_relay_for_duration(duration)
        self.shutdown_relay()

    def door_backward(self, duration: float):
        """
        This method moves the incubator door backward.

        Args:
            duration: The duration for which the relay should be on.
        """
        GPIO.output(self.relay0, GPIO.LOW)
        GPIO.output(self.relay1, GPIO.HIGH)
        self._run_relay_for_duration(duration)
        self.shutdown_relay()

    def shutdown_relay(self):
        """
        This method shuts down the relay.
        """
        GPIO.output(self.relay0, GPIO.HIGH)
        GPIO.output(self.relay1, GPIO.HIGH)

    def shutdown(self):
        """
        This method shuts down the incubator door controller.
        """
        self.logger.info("Shutting down the incubator")
        self._is_shutting_down = True
        self.shutdown_relay()

    def _run_relay_for_duration(self, duration: float):
        """
        Helper method to run the relay for a given duration,
        checking for shutdown.

        Args:
            duration: The duration for which the relay should be on.
        """
        end_time = time.time() + duration
        while time.time() < end_time:
            if self._is_shutting_down:
                self.logger.info("Operation stopped due to shutdown")
                break
            time.sleep(0.1)  # Check every 100ms
        self.shutdown_relay()


incubator_api = IncubatorAPI()
