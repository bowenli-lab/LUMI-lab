from typing import Optional

from .base_template import BaseMessageTemplate


class OpentronProgressTemplate(BaseMessageTemplate):
    """
    This is a template class for the Opentron progress template.
    """

    def __init__(self,
                 experiment_id: str,
                 task_name: str,
                 progress: str,
                 elapsed_time: Optional[str] = None,
                 *args, **kwargs):
        self.experiment_id = experiment_id
        self.progress = progress
        self.elapsed_time = elapsed_time
        self.task_name = task_name

    def parse(self,) -> str:
        """
        This method parses the message based on the template.
        """
        return_string = f"""
        
        * 🔬 Experiment ID:* `{self.experiment_id}`
        * 📝 Task:* `{self.task_name}`
        * 📊 Progress:* `{self.progress}`
        
        """

        if self.elapsed_time:
            return_string += f"* ⏰ Elapsed Time:* `{self.elapsed_time}`"

        return return_string
