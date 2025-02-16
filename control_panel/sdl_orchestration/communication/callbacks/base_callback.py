from typing import Dict, List, Optional

from sdl_orchestration.communication.device_controller import \
    BaseDeviceController


class CallbackState:
    finish: str = "finish"
    pause: str = "pause"
    play: str = "play"
    error: str = "ERROR"


class BaseCallback:

    """
    This is the base callback class. It is used to create a callback object
    that can be used to determine the action and return a string as the
    instruction.

    This is a Callable class.
    """

    def __call__(self,
                 variables_query_dict: Dict[str, str],
                 ) -> str:
        """
        Base callback method is used to determine the action and return a
        string as the instruction.

        Args:
            variables_query_dict (Dict[str, str]): The variables query dict.

        """
        raise NotImplementedError("This method needs to be implemented.")
