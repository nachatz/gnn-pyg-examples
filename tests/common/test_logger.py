import logging

from common.logger import configure_logging
from common.logger import logger


class TestConfigureLogging:
    def test_installs_one_bare_message_handler(self) -> None:
        configure_logging()
        configure_logging()
        (handler,) = logger.handlers
        assert handler.formatter is not None
        assert handler.formatter.format(logging.makeLogRecord({"msg": "hi"})) == "hi"
