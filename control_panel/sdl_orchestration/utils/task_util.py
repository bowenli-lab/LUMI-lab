import os
from typing import Any, List, Optional

from bson.objectid import ObjectId


def get_task_id_from_lst(lst: List[Any]):
    """
    Get task_id from a list of tasks.
    """
    res = [item.task_id for item in lst]
    return res


def get_full_path_for_file(file_name: str, dir_lst: List[str],
                           add_ext: Optional[str] = ".py"
                           ) -> Optional[str]:
    """
    Get the full path for a file by joining the file name with the directory list
    and checking if the file exists.

    Args:
        file_name (str): The name of the file.
        dir_lst (List[str]): The list of directories to check.
        add_ext (Optional[str], optional): The extension to add to the file name.

    Returns:
        Optional[str]: The full path of the file if it exists, otherwise None.
    """
    if add_ext and add_ext not in file_name:
        file_name += add_ext

    for dir_name in dir_lst:
        full_path = f"{dir_name}/{file_name}"
        if os.path.exists(full_path):
            return full_path
    return None
