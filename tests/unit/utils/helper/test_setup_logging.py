import logging
import sys

import pytest

from power_quant_strategies.utils.helper import setup_logging as module


@pytest.fixture
def clean_root_logger():
    """Give each test an unconfigured root logger and restore the old state afterwards."""
    root = logging.getLogger()
    old_handlers, old_level, old_hook = root.handlers[:], root.level, sys.excepthook
    root.handlers = []
    yield root
    for handler in root.handlers:
        handler.close()
    root.handlers, sys.excepthook = old_handlers, old_hook
    root.setLevel(old_level)


def test_writes_log_file_and_console(tmp_path, clean_root_logger):
    # basicConfig does nothing while the root logger has handlers; pytest adds one for each test
    clean_root_logger.handlers = []
    save_path = tmp_path / "logs"
    module.setup_logging(save_path=str(save_path))

    logging.getLogger().info("hello from the test")
    for handler in clean_root_logger.handlers:
        handler.flush()

    assert clean_root_logger.level == logging.DEBUG
    assert {type(h) for h in clean_root_logger.handlers} == {logging.FileHandler, logging.StreamHandler}
    assert "hello from the test" in (save_path / "app.log").read_text()


def test_uncaught_exceptions_are_logged(tmp_path, clean_root_logger, caplog):
    module.setup_logging(save_path=str(tmp_path))
    assert sys.excepthook is module._excepthook

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        with caplog.at_level(logging.ERROR):
            sys.excepthook(*sys.exc_info())

    assert "Uncaught exception" in caplog.text
    assert "boom" in caplog.text
