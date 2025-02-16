import RPi.GPIO as GPIO
import board
from adafruit_motorkit import MotorKit
import logging
import os

CLAMP_STEPS = 60
DIRECTION_IN = -1
DIRECTION_OUT = 1


class ClampStatus:
    """
    The status of the motor.
    """
    CLOSED = 1
    OPEN = 0


class ClampAPI:
    def __init__(
            self,
    ):
        self._is_shutting_down = False
        self.number_of_motors = 1
        self.kit_list = []
        self.motor_list = []
        self.plate_position = 0
        self.logger = self._config_logger()
        self.control_address = 0x64
        self.motor_pos = 0
        self._init_motors(self.number_of_motors)
        self.motor = self.motor_list[0]
        self.clamp_status = ClampStatus.OPEN
        self.current_position = 0

    def _init_motors(self, number: int) -> None:
        """
        This method initializes the motor_list and kit_list.
        """
        # initialize the kit list
        for kit_num in range(1):
            self.kit_list.append(MotorKit(i2c=board.I2C(),
                                          address=self.control_address +
                                                  kit_num))

        # initialize the motor list
        for kit in self.kit_list:
            self.motor_list.extend([kit.stepper1, ])

        # remove the extra motors
        self.motor_list = self.motor_list[:number]

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
            f"./logs/clamp.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if __name__ == "__main__":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def clamp(self, ):
        """
        This method clamps the plate.
        """
        if self.clamp_status == ClampStatus.CLOSED:
            self.logger.info("Plate is already clamped.")
            return "Warning: Plate is already clamped."
        self.logger.info("Clamping the plate.")
        self._move_motor(steps=CLAMP_STEPS, direction=DIRECTION_IN)
        self.clamp_status = ClampStatus.CLOSED
        self.logger.info("Plate clamped. Locking the motors.")
        return "Plate clamped."

    def release(self, ):
        """
        This method releases the plate.
        """
        if self.clamp_status == ClampStatus.OPEN:
            self.logger.info("Plate is already released.")
            return "Warning: Plate is already released."
        self.logger.info("Releasing the plate.")
        self._move_motor(steps=CLAMP_STEPS, direction=DIRECTION_OUT)
        self.clamp_status = ClampStatus.OPEN
        self.logger.info("Plate released. Unlocking the motors.")
        self.release_motors()
        return "Plate released."

    def reset_clamped(self):
        """
        This method resets the clamped status.
        """
        self.clamp_status = ClampStatus.CLOSED
        return "Clamped status reset to closed."

    def move_in(self, steps: int):
        """
        This method moves the plate in by the given number of steps.
        """
        self.logger.info(f"Moving the plate in by {steps} steps.")
        self._move_motor(steps=steps, direction=DIRECTION_IN)
        self.release_motors()

    def move_out(self, steps: int):
        """
        This method moves the plate out by the given number of steps.
        """
        self.logger.info(f"Moving the plate out by {steps} steps.")
        self._move_motor(steps=steps, direction=DIRECTION_OUT)
        self.release_motors()

    def _move_motor(self, steps: int, direction: int):
        """
        This method moves the motor by the given number of steps.
        """
        self.logger.info(f"Moving the motor by {steps} steps.")
        for _ in range(steps):
            self.motor.onestep(direction=direction)

    def release_motors(self):
        """
        This method releases the motors.
        """
        for motor in self.motor_list:
            motor.release()

    def shutdown(self):
        self.logger.info("Shutting down the clamp")
        self._is_shutting_down = True
        self.release_motors()


clamp_api = ClampAPI()
