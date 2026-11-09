import logging
import logging.handlers

from ml import config


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Console + rotating file logging. No patient data or raw filenames are logged."""
    logger = logging.getLogger("neurovision")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        config.ensure_dirs()
        fh = logging.handlers.RotatingFileHandler(config.RESULTS_DIR / "backend.log", maxBytes=1_000_000, backupCount=2)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass
    return logger
