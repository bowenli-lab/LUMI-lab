from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from bson import ObjectId


class DeviceStatus(Enum):
    """
    The status of the device.
    """
    RUNNING = "RUNNING"
    IDLE = "IDLE"
    ERROR = "ERROR"
    DISCONNECTED = "DISCONNECTED"


class BaseDeviceController(ABC):
    """
    The abstract class of device.

    All the devices should be inherited from this class
    """
    status: DeviceStatus = DeviceStatus.DISCONNECTED

    def __init__(self, device_id: ObjectId = ObjectId(), device_name: str = ""):
        self.device_id = device_id
        self.status = DeviceStatus.DISCONNECTED
        self.device_name = device_name

    @abstractmethod
    def run(self, program: Any, task_id: Optional[ObjectId] = None,
            experiment_id: Optional[ObjectId] = None) -> None:
        raise NotImplementedError("run method is not implemented")

    @abstractmethod
    def connect(self) -> None:
        """
        Connect to any devices here. This will be called to make connections
        to devices at the appropriate time.

        This method must be defined even if no device connections are
        required! Just return in this case.
        """
        raise NotImplementedError()

    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from devices here. TThis will be called to make
        connections to devices at the appropriate time.

        This method must be defined even if no device connections are
        required! Just return in this case.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_running(self) -> bool:
        """Check whether this device is running."""
        raise NotImplementedError()

    @abstractmethod
    def is_idle(self) -> bool:
        """Check whether this device is idle."""
        raise NotImplementedError()
