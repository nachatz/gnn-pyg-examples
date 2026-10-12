import logging
import sys

from common.config import settings

logger = logging.getLogger("gnn_pyg_examples")


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.setLevel(settings.log_level)
