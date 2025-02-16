
def parse_human_readable_time(seconds: int) -> str:
    """
    This function converts seconds to human-readable time format.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours == 0:
        return f"{minutes} minutes, {seconds} seconds"
    else:
        return f"{hours} hours, {minutes} minutes, {seconds} seconds"
