import datetime

DEFAULT_TIME_FORMAT = "%Y%m%d%H%M%S"


def get_timestamp(time_format=DEFAULT_TIME_FORMAT):
    return datetime.datetime.now().strftime(time_format)
