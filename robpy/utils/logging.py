import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Parameters:
        name (str): The name of the logger.
        level (int): The logging level.

    Returns:
        logger: A logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    logger.addHandler(handler)
    return logger
