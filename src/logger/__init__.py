"""
Initializes a global logger using a configuration file.
"""

LOGGER = None


def init():
    """Reads a `logging.ini` file from the same directory as the module, sets up the logging configuration, and assigns
    the global `LOGGER` object."""
    import logging
    from logging.config import fileConfig
    from pathlib import Path

    global LOGGER

    logger_config_file = Path(__file__).parent.joinpath('logging.ini').resolve()
    if not logger_config_file.exists():
        raise FileNotFoundError('Missing logger configuration file.')
    fileConfig(logger_config_file)
    LOGGER = logging.getLogger()
    LOGGER.debug(f'Loaded LOGGER from {logger_config_file}')


init()
