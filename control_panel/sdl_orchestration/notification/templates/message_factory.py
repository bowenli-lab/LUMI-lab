from sdl_orchestration.notification.templates.error_template import \
    ErrorTemplate
from sdl_orchestration.notification.templates.opentron_progress_template import \
    OpentronProgressTemplate


class MessageFactory:
    """
    Factory class to create messages based on the message type
    """
    @staticmethod
    def create_message(message_type, *args, **kwargs):
        if message_type == "progress":
            return OpentronProgressTemplate(*args, **kwargs)
        elif message_type == "reagent":
            pass
        elif message_type == "result":
            pass
        elif message_type == "error":
            return ErrorTemplate(*args, **kwargs)
        else:
            raise NotImplementedError
