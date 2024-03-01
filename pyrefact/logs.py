"""Logging"""

import functools
import logging


class _Message:
    def __init__(self, /, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.fmt.format(*self.args, **self.kwargs)


@functools.lru_cache(maxsize=1)
def _get_logger() -> logging.Logger:
    # Why is this so complicated

    logger = logging.getLogger("pyrefact")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def info(fmt, /, *args, **kwargs):
    return _get_logger().info(_Message(fmt, *args, **kwargs))


def debug(fmt, /, *args, **kwargs):
    return _get_logger().debug(_Message(fmt, *args, **kwargs))


def error(fmt, /, *args, **kwargs):
    return _get_logger().error(_Message(fmt, *args, **kwargs))


def set_level(level: int) -> None:
    _get_logger().setLevel(level)
