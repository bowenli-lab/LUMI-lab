import csv
import time
from queue import PriorityQueue
from typing import List
import board
from adafruit_motorkit import MotorKit

ASPIRATE_ADJUSTMENT = 0.3


class MotorStatus:
    """
    The status of the motor.
    """
    STOP = 0
    FORWARD = 1
    BACKWARD = -1


class LiquidSamplerConstants:
    """
    Constants for the liquid sampler.
    """
    PLATE_MOVE_TIME = 10
    PUMP_TIME = 5  # base constant for the pump time
    LEAD_SCREW_POSITION = 0

    NUM_PUMPS = 96
    NUM_LEAD_SCREWS = 1

    MAX_MOTOR_ACTIVATED_PER_KIT = 2
    MAX_MOTOR_ACTIVATED = 6


def read_motor_config(filename):
    """
    Read motor configuration from a CSV file
    and store as a list of dictionaries.
    """
    config = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            config.append(row)
    return config


class LiquidSamplerAPI:
    def __init__(self, ):
        self.number_of_motors = (LiquidSamplerConstants.NUM_LEAD_SCREWS +
                                 LiquidSamplerConstants.NUM_PUMPS)
        self.motor_list = []
        self.motor_status_list = [MotorStatus.STOP] * self.number_of_motors

        self.motor_config = read_motor_config("General_mapping_sampler.csv")

        self._init_kit_dict()
        self._init_motors()

    def _init_kit_dict(self, ):
        """
        This method initializes the kit dictionary.
        """
        self.kit_dict = {}
        for row in self.motor_config:
            motor_kit_num = int(row["DBoard_ID"])
            if motor_kit_num not in self.kit_dict:
                self.kit_dict[motor_kit_num] = MotorKit(i2c=board.I2C(),
                                                        address=0x60 +
                                                                motor_kit_num)

    def _init_motors(self, ) -> None:
        """
        This method initializes the motor_list and kit_list.
        """
        self.motor_dict = {}
        for row in self.motor_config:
            kit_num = int(row["DBoard_ID"])
            motor_num = int(row["Motor_ID"])
            motor_on_board_num = int(row["DBoard_cor"])
            print(
                f"kit_num: {kit_num}, motor_num: {motor_num}, "
                f"motor_on_board_num: {motor_on_board_num}")
            motor = self.kit_dict[kit_num].__getattribute__(
                f"motor{motor_on_board_num}")
            print(f"motor: {motor}")
            self.motor_dict[motor_num] = motor

    def _update_motor_status(self,
                             motor_number: int,
                             status: int):
        """
        This method updates the motor status.

        Send command to the motor to move in the specified direction and track
        the status of the motor.
        """
        try:
            self.motor_status_list[int(motor_number)] = status
            print(
                f"updating motor_number: {motor_number}, {self.motor_dict[int(motor_number)]}")
            self.motor_dict[int(motor_number)].throttle = status
        except KeyboardInterrupt:
            self.stop_all()

    def plate_in(self, ):
        """
        This method moves the plate in.
        """
        self._update_motor_status(
            LiquidSamplerConstants.LEAD_SCREW_POSITION,
            MotorStatus.BACKWARD)
        time.sleep(LiquidSamplerConstants.PLATE_MOVE_TIME)
        self._update_motor_status(
            LiquidSamplerConstants.LEAD_SCREW_POSITION,
            MotorStatus.STOP)

    def plate_out(self, ):
        """
        This method moves the plate out.
        """
        self._update_motor_status(
            LiquidSamplerConstants.LEAD_SCREW_POSITION,
            MotorStatus.FORWARD)
        time.sleep(LiquidSamplerConstants.PLATE_MOVE_TIME)
        self._update_motor_status(
            LiquidSamplerConstants.LEAD_SCREW_POSITION,
            MotorStatus.STOP)

    def _pump(self, motor_number: int, amount: float = 1.0):
        """
        This method pumps liquid at the specified motor number and amount.

        The amount is the coefficient of the pump time.
        """

        self._update_motor_status(motor_number, MotorStatus.BACKWARD)
        time.sleep(LiquidSamplerConstants.PUMP_TIME * amount)
        self._update_motor_status(motor_number, MotorStatus.STOP)

    def _aspirate(self, motor_number: int, amount: float = 1.0):
        """
        This method pumps liquid at the specified motor number and amount.

        The amount is the coefficient of the pump time.
        """

        self._update_motor_status(motor_number, MotorStatus.FORWARD)
        time.sleep(LiquidSamplerConstants.PUMP_TIME * amount)
        self._update_motor_status(motor_number, MotorStatus.STOP)

    def pump(self, motor_number: int, amount: float = 1.0):
        """
        This method pumps liquid at the specified motor number and amount.
        The method will aspirate a small amount of liquid before pumping and
        after pumping to prevent the liquid from dripping.
        """
        self._aspirate(motor_number, amount * ASPIRATE_ADJUSTMENT)
        self._pump(motor_number, amount)
        self._aspirate(motor_number, amount)

    def aspirate(self, motor_number: int, amount: float = 1.0):
        """
        This method aspirates liquid at the specified motor number and amount.
        """
        self._aspirate(motor_number, amount)

    def _batch_operation(self, motor_number_list: List[int],
                         amount_list: List[float],
                         direction: int):
        """
        This method performs a batch operation on the motors.
        """
        priority_queue = PriorityQueue()

        pump_time_list = [LiquidSamplerConstants.PUMP_TIME * amount
                          for amount in amount_list]

        # initialize the clock list
        clock_list = {
            idx: 0.0
            for idx in range(len(motor_number_list))
        }

        # initialize the priority queue
        for idx, motor_number in enumerate(motor_number_list):
            priority_queue.put((pump_time_list[idx], (motor_number, idx)))

        # start the pumping
        for idx, motor_number in enumerate(motor_number_list):
            self._update_motor_status(motor_number, direction)
            clock_list[idx] = time.time()

        # end the pumping
        while not priority_queue.empty():
            # get the top element
            top_element = priority_queue.get()
            motor_number, idx = top_element[1]

            # check if the clock is at the maximum
            if time.time() - clock_list[idx] >= pump_time_list[idx]:
                self._update_motor_status(motor_number, MotorStatus.STOP)
            else:
                # re queue
                priority_queue.put((pump_time_list[idx], (motor_number, idx)))

    def _generate_safe_mini_batch(self,
                                  motor_number_list: List[int],
                                  amount_list: List[float],
                                  limitation_per_kit=LiquidSamplerConstants.MAX_MOTOR_ACTIVATED_PER_KIT,
                                  max_motor=LiquidSamplerConstants.MAX_MOTOR_ACTIVATED) -> List:
        """
        Generate mini-batches of motor operations based on the constraints of
        the number of simultaneous operations both per kit and max motor limit.
        """
        # Map each motor to its corresponding kit using motor configuration
        motor_to_kit = {int(row["Motor_ID"]): int(row["DBoard_ID"]) for row in
                        self.motor_config}

        # Organize motors by kit
        kit_to_motors = {}
        for motor in motor_number_list:
            kit = motor_to_kit[motor]
            if kit not in kit_to_motors:
                kit_to_motors[kit] = []
            idx = motor_number_list.index(motor)
            kit_to_motors[kit].append((motor, amount_list[idx]))

        # Prepare batches respecting the kit limitation and the max motor count
        batches = []
        current_batch = []
        current_batch_kit_count = {}

        for kit, motors in kit_to_motors.items():
            for motor, amount in motors:
                kit_count = current_batch_kit_count.get(kit, 0)
                if (len(current_batch) >= max_motor or kit_count >=
                        limitation_per_kit):
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_kit_count = {}

                current_batch.append((motor, amount))
                current_batch_kit_count[kit] = (
                        current_batch_kit_count.get(kit, 0) + 1)

        # Add the last batch if not empty
        if current_batch:
            batches.append(current_batch)

        print(f"Generated mini-batches: {batches}")

        return batches

    def _safe_batch_operation(self, motor_number_list: List[int],
                              amount_list: List[float],
                              direction: int,
                              limitation_per_kit=LiquidSamplerConstants.MAX_MOTOR_ACTIVATED_PER_KIT,
                              max_motor=LiquidSamplerConstants.MAX_MOTOR_ACTIVATED):
        """
        This method performs a batch operation on the motors. Due to the voltage
        limitation on the board, we can only run a certain number of motors on a
        same kit at the same time.

        The method will limit the number of motors running on the same kit at
        the same time.

        This method also warm up the motors before the operation.

        """
        # Generate safe mini-batches
        mini_batches = self._generate_safe_mini_batch(motor_number_list,
                                                      amount_list,
                                                      limitation_per_kit,
                                                      max_motor)

        # Start the mini-batches
        for idx, mini_batch in enumerate(mini_batches):
            print(f"Starting {idx + 1}/{len(mini_batches)} mini-batch.")
            print(f"Content: {mini_batch}")
            motor_numbers = [motor for motor, _ in mini_batch]
            amounts = [amount for _, amount in mini_batch]
            aspirate_adj_amounts = [amount * ASPIRATE_ADJUSTMENT
                                    for amount in amounts]

            # # warm up the motors
            # self.batch_warmup(motor_numbers, 0.5)

            # self._batch_operation(motor_numbers, aspirate_adj_amounts,
            #                       -direction)
            self._batch_operation(motor_numbers, amounts, direction)
            # self._batch_operation(motor_numbers, amounts, -direction)

    def batch_warmup(self, motor_number_list: List[int],
                     duration: float):
        """
        This method warms up the motors.
        """
        for _ in range(1):
            self._batch_operation(motor_number_list,
                                  [duration] * len(motor_number_list),
                                  MotorStatus.BACKWARD)
            self._batch_operation(motor_number_list,
                                  [duration] * len(motor_number_list),
                                  MotorStatus.FORWARD)

    def batch_pump(self, motor_number_list: List[int],
                   amount_list: List[float]):
        """
        This method pumps liquid at the specified motor batch and amount, at
        the same time.

        The pumping time is managed by tracking the pumping time.
        """
        self._safe_batch_operation(motor_number_list,
                                   amount_list,
                                   MotorStatus.BACKWARD)

    def batch_aspirate(self, motor_number_list: List[int],
                       amount_list: List[float]):
        """
          This method aspirates liquid at the specified motor batch and amount, at
          the same time.

          The pumping time is managed by tracking the pumping time.
          """
        self._safe_batch_operation(motor_number_list,
                                   amount_list,
                                   MotorStatus.FORWARD)

    def calibrate(self, ):
        """
        This method calibrates the motors.
        """
        # TODO: implement the calibration method
        raise NotImplementedError

    def stop_all(self, ):
        """
        This method stops all motors.
        """
        for motor_number in range(1, self.number_of_motors):
            self._update_motor_status(motor_number, MotorStatus.STOP)

    def shutdown(self, ):
        """
        This method shuts down the motors.
        """
        print("Shutting down the motors.")
        self.stop_all()


liquid_sampler_api = LiquidSamplerAPI()
