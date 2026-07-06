import logging
import sys
from logging.handlers import RotatingFileHandler
from settings.Define import PathConfig


def setup_logger(name: str, log_file: str = None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    PathConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = PathConfig.LOG_DIR / "app.log"
    else:
        log_file = PathConfig.LOG_DIR / log_file

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = None):
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger