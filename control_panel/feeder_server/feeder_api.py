from typing import Optional
import RPi.GPIO as GPIO
import board
from adafruit_motorkit import MotorKit
import logging
import sys
import os
import time

CONFIG_FILE = "./feeder_mapping.csv"

PLATE_BASE_POSITION = 0  # in steps
HALF_SOUND_SPEED = 17150  # in cm/s
TIP_RACK_HEIGHT = 1550  # in steps
PCR_DEEPWELL_HEIGHT = 1050  # in steps
WHITE_PLATE_HEIGHT = 319  # in steps
DEEPWELL_W_COVER_HEIGHT = 1735  # in steps


def read_config(file_path: str):
    # Reading the CSV file
    with open(file_path, "r") as file:
        content = file.readlines()

    # Splitting the header and the rest of the data
    header = content[0].strip().split(",")
    rows = [line.strip().split(",") for line in content[1:]]

    # Creating the dictionary
    data_dict = {
        int(row[0]): {header[i]: row[i] for i in range(1, len(header))} for row
        in rows
    }
    return data_dict


def _sensor_cleanup():
    """
    This function cleans up the sensor.
    """
    GPIO.cleanup()


class MotorStatus:
    """
    The status of the motor.
    """

    STOP = 0
    FORWARD = -1
    BACKWARD = 1


class FeederAPI:
    heights = [
        TIP_RACK_HEIGHT,
        TIP_RACK_HEIGHT,
        PCR_DEEPWELL_HEIGHT,
        WHITE_PLATE_HEIGHT,
    ]

    feeder_specific_config = {
        2: {
            "PCR_DEEPWELL": PCR_DEEPWELL_HEIGHT,
            "DEEPWELL_W_COVER": DEEPWELL_W_COVER_HEIGHT
        }
    }

    def __init__(
            self,
            device_code: int,
            use_sensor: bool = False,
    ):
        self.number_of_motors = 2
        self.kit_list = []
        self.motor_list = []
        self.motor_status_list = [MotorStatus.STOP] * self.number_of_motors
        self.plate_position = 0
        self.device_code = device_code
        self.logger = self._config_logger()
        self.config = read_config(CONFIG_FILE)[self.device_code]
        self.control_address = int(self.config["address"], 16)
        self.trig = int(self.config["TRIG"])
        self.echo = int(self.config["ECHO"])
        self.pos_adj_constant = int(self.config["POS_ADJ_CONST"])
        self._cargo_height = FeederAPI.heights[self.device_code]  # in steps
        if int(self.device_code) in FeederAPI.feeder_specific_config:
            # Specific configuration for the feeder
            self._cargo2height = FeederAPI.feeder_specific_config[
                int(self.device_code)]
        else:
            self._cargo2height = None
        self.use_sensor = use_sensor
        if use_sensor:
            self._init_sensor()
        self._init_motors(self.number_of_motors)
        self.sensor_threshold = int(self.config["SENSOR_THRESHOLD"])
        self._is_shutting_down = False
        self._is_running = False
        self._log_feeder_config()
        self.logger.info("Feeder API initialized.")

    @property
    def cargo_height(self):
        return self._cargo_height

    @cargo_height.setter
    def cargo_height(self, value: int):
        self._cargo_height = int(value)
        self.logger.info(f"Setting the cargo height to {self._cargo_height}.")

    def _log_feeder_config(self):
        """
        This function prints the feeder configuration.
        """
        self.logger.info(f"Feeder configuration: {self.config}")

    def _init_motors(self, number: int) -> None:
        """
        This method initializes the motor_list and kit_list.
        """

        # initialize the kit list
        for kit_num in range(1):
            self.kit_list.append(
                MotorKit(i2c=board.I2C(),
                         address=self.control_address + kit_num)
            )

        # initialize the motor list
        for kit in self.kit_list:
            self.motor_list.extend([kit.stepper1, kit.stepper2])

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
            f"./logs/feeder{self.device_code}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        if __name__ == "__main__":
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

    def move_up(self, steps: int, speed: float = 1.0):
        """
        This method moves the both motors up.
        """
        self.logger.info(
            f"Before moving; Current plate position: " f"{self.plate_position}."
        )

        stepper1 = self.motor_list[0]
        stepper2 = self.motor_list[1]

        steps = self._check_steps(steps, MotorStatus.FORWARD)
        self.move_steppers(stepper1, stepper2, steps, MotorStatus.FORWARD,
                           speed)
        self.plate_position += steps
        self._release_motors()
        self.logger.info(
            f"After moving; Current plate position: {self.plate_position}."
        )

    def move_down(self, steps: int, speed: float = 1.0):
        """
        This method moves both motors down.
        """
        self.logger.info(f"Current plate position: {self.plate_position}.")

        stepper1 = self.motor_list[0]
        stepper2 = self.motor_list[1]

        steps = self._check_steps(steps, MotorStatus.BACKWARD)

        self.move_steppers(stepper1, stepper2, steps, MotorStatus.BACKWARD,
                           speed)
        self.plate_position -= steps
        self._release_motors()
        self.logger.info(f"Current plate position: {self.plate_position}.")

    def move_down_cargo(self, cargo_num: int, speed: float = 1.0):
        """
        This method moves the plate down by the cargo height.

        Args:
            cargo_num: The number of cargos to move down.
            speed: The speed of the motors.
        """
        self.move_down(self.cargo_height * cargo_num, speed)

    def move_up_cargo(self, cargo_num: int, speed: float = 1.0):
        """
        This method moves the plate up by the cargo height.

        Args:
            cargo_num: The number of cargos to move up.
            speed: The speed of the motors.
        """
        self.move_up(self.cargo_height * cargo_num, speed)

    def _verify_cargo_type(self, cargo_type: str) -> bool:
        """
        This method verifies the cargo type.
        """
        if self._cargo2height is None:
            self.logger.error("Cargo to height mapping is not provided.")
            return False

        if cargo_type not in self._cargo2height:
            self.logger.error(f"Invalid cargo type: {cargo_type}.")
            return False

        return True

    def move_up_cargo_by_type(self, cargo_type: str,
                              cargo_num: int,
                              speed: float = 1.0) -> bool:
        """
        This method moves the plate up by cargo type and number.

        Args:
            cargo_type: The type of cargo to move up.
            cargo_num: The number of cargos to move up.
            speed: The speed of the motors.

        Returns:
            bool: True if the operation is successful, False otherwise.
        """
        if not self._verify_cargo_type(cargo_type):
            return False

        self.move_up(self._cargo2height[cargo_type] * cargo_num, speed)

        return True

    def move_down_cargo_by_type(self, cargo_type: str,
                                cargo_num: int,
                                speed: float = 1.0) -> bool:
        """
        This method moves the plate down by cargo type and number.

        Args:
            cargo_type: The type of cargo to move down.
            cargo_num: The number of cargos to move down.
            speed: The speed of the motors.
        Returns:
            bool: True if the operation is successful, False otherwise.
        """

        if not self._verify_cargo_type(cargo_type):
            return False

        self.move_down(self._cargo2height[cargo_type] * cargo_num, speed)

        return True

    def move_bottom(self, speed: float = 1.0):
        """
        This method moves the plate to the bottom.
        """
        self.move_to_position(PLATE_BASE_POSITION, speed)

    def move_to_position(self, position: int, speed: float = 1.0):
        """
        This method moves the plate to the specified position.
        """

        steps = position - self.plate_position

        self.logger.info(
            f"Before moving; Current plate position: {self.plate_position}."
        )
        self.logger.info(f"Moving the plate by {steps} steps.")

        if steps > 0:
            self.move_up(steps, speed)
        else:
            self.move_down(abs(steps), speed)

        self.logger.info(
            f"After moving; Current plate position: {self.plate_position}."
        )

        self._release_motors()

    def feed_plate_auto(self, move_step: int = 10):
        """
        This method moves the plate up to feed the sensor.
        Assume: sensor now detects no object.
        We rise the plate up until the sensor detects an object.

        """
        self.logger.info("Feeding the plate.")
        while not self.query_distance():
            self.move_up(move_step)
        # when we detect the object, we move the plate up a bit
        # self.move_up(self.pos_adj_constant)
        self._release_motors()

    def feed_plate(self, num_of_cargos: Optional[int] = 1):
        """
        This method moves the plate up and feeds the cargos out. For each cargo,
        it moves it up the number of steps based on :attr:`self.cargo_height`.

        Args:
            num_of_cargos: The number of cargos to feed. If None, it feeds all the
                cargos one by one.
        """
        self.logger.info("Feeding the plate.")
        if num_of_cargos is None:
            raise NotImplementedError
            # num_of_cargos = self.plate_position // self.cargo_height

        for _ in range(num_of_cargos):
            self.move_up(self.cargo_height)
            # self.wait_for_ready()
            time.sleep(10)
        self._release_motors()

    def is_running(self):
        """
        This method returns whether the feeder is running.
        """
        return self._is_running

    def query_distance(
            self,
            try_count: int = 4,
    ) -> bool:
        """
        This method queries the distance from the sensor, with a median filter.
        """
        if not self.use_sensor:
            raise ValueError("Sensor is not enabled.")
        distances = []
        for _ in range(try_count):
            read_out = self.sensor_read()
            if read_out == -1:
                self.logger.warning(
                    "Error in reading the sensor, skipping this reading."
                )
                continue
            distances.append(read_out)

        if not distances:
            raise ValueError("All sensor readings failed.")

        # Median filtering to handle outliers
        distances.sort()
        median_distance = distances[len(distances) // 2]

        # Calculate the average of the valid distances
        average_distance = sum(distances) / len(distances)

        self.logger.info(
            f"Distance (avg): {round(average_distance, 3)} cm."
            f" Median: {round(median_distance, 3)} cm."
        )

        detect = median_distance < self.sensor_threshold
        if detect:
            self.logger.info(
                f"Object detected at distance: " f"{round(median_distance, 3)} cm."
            )

        return detect

    def _release_motors(self):
        """
        This method releases the motors break.
        """
        for motor in self.motor_list:
            motor.release()

    def release_motors(self):
        """
        This method releases the motors break.
        """
        self._release_motors()

    def _check_steps(self, steps: int, direction: int) -> int:
        """
        This method checks if the steps are within the limits.
        """
        # disable this check
        if (
                direction == MotorStatus.BACKWARD
                and False
                and ((self.plate_position - steps) < PLATE_BASE_POSITION)
        ):
            self.logger.info(
                f"Cannot move the plate further down, adjusting the steps to"
                f" {self.plate_position - PLATE_BASE_POSITION}"
            )
            return self.plate_position - PLATE_BASE_POSITION
        else:
            return steps

    def _init_sensor(
            self,
    ):
        """
        This function initializes the sensor.
        """
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig, GPIO.OUT)
            GPIO.setup(self.echo, GPIO.IN)
        except Exception as e:
            self.logger.error(f"Error in initializing the sensor: {e}")
            sys.exit(1)
        self.logger.info(
            f"Sensor initialized. trig: {self.trig}, " f"echo: {self.echo}."
        )

    def sensor_read(self, timeout=2):
        """
        This function reads the distance from the sensor.
        """
        trig, echo = self.trig, self.echo
        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)

        pulse_start = None
        pulse_end = None

        wait_start = time.time()
        while GPIO.input(echo) == 0:
            pulse_start = time.time()
            if time.time() - wait_start > timeout:
                self.logger.warning("Timeout in reading the sensor. Returning "
                                    "-1.")
                return -1

        wait_start = time.time()
        while GPIO.input(echo) == 1:
            pulse_end = time.time()
            if time.time() - wait_start > timeout:
                self.logger.warning("Timeout in reading the sensor. Returning "
                                    "-1.")
                return -1

        if pulse_start is None or pulse_end is None:
            self.logger.warning("Error in reading the sensor. Returning -1.")
            return -1

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * HALF_SOUND_SPEED

        return distance

    def move_steppers(self, stepper1, stepper2, steps, direction, speed=1.0):
        """
        This function moves both stepper motors simultaneously.
        """
        try:
            self._is_running = True
            for _ in range(abs(steps)):
                if self._is_shutting_down:
                    self.logger.info(
                        "Shutting down detected. Stopping and releasing the "
                        "motors."
                    )
                    break
                stepper1.onestep(direction=direction)
                stepper2.onestep(direction=direction)
            self._is_running = False
        except KeyboardInterrupt:
            self.logger.info(
                "Keyboard interrupt detected. Stopping and releasing the "
                "motors."
            )
            stepper1.release()
            stepper2.release()
            _sensor_cleanup()
        finally:
            stepper1.release()
            stepper2.release()

    def shutdown(self):
        self.logger.info("Shutting down the feeder.")
        self._is_shutting_down = True
        self._release_motors()
        _sensor_cleanup()


if __name__ == "__main__":
    _sensor_cleanup()
    feeder = FeederAPI(0)
    while True:
        time.sleep(0.5)
        print(feeder.query_distance())
