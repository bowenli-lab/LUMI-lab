from typing import Optional
from .base_template import BaseMessageTemplate


class ErrorTemplate(BaseMessageTemplate):
    """
    Error template class
    """
    def __init__(self, error_message, *args, **kwargs):
        self.error_message = error_message
        self.other_info = args

    def parse(self,) -> str:
        """
        This method parses the message based on the template.
        """
        return_string = f"""
        
        * 🚨 Error:* `{self.error_message}`
        """

        if self.other_info:
            return_string += "\n\n"
            for info in self.other_info:
                return_string += f"*{info}*\n"

        return return_string
