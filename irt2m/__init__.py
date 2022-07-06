# -*- coding: utf-8 -*-

"""IRT2M import-time things."""


import logging
import logging.config
import os
from enum import Enum
from pathlib import Path

import yaml
from ktz.filesystem import path as kpath

_root_path = kpath(__file__).parent.parent

#
# ENV VARS
#

ENV_DIR_DATA = "IRT2M_DATA"

ENV_LOG_CONF = "IRT2M_LOG_CONF"
ENV_LOG_FILE = "IRT2M_LOG_FILE"
ENV_LOG_FILE_LIBS = "IRT2M_LOG_FILE_LIBS"


# check whether data directory is overwritten
if ENV_DIR_DATA in os.environ:
    _data_path = kpath(os.environ[ENV_DIR_DATA])
else:
    _data_path = kpath(_root_path / "data", create=True)


#
# CONFIGURATION
#


class _DIR:

    ROOT: Path = _root_path
    DATA: Path = _data_path
    CONF: Path = kpath(_root_path / "conf", create=True)
    CACHE: Path = kpath(_data_path / "cache", create=True)


class ENV:
    """IRT2M environment."""

    DIR = _DIR


class IRT2Error(Exception):
    """General error."""

    pass


# special tokenizer tokens
# TODO move to irt2m.data
class TOKEN(Enum):
    """Special tokens used by the BERT Tokenizer."""

    mask = "[MASK]"
    mention_start = "[MENTION_START]"
    mention_end = "[MENTION_END]"

    @classmethod
    def values(Self):
        """Get a list of token identifier."""
        return [e.value for e in Self]


#
# LOGGING
#


log = logging.getLogger(__name__)

# if used as library do not log anything
log.addHandler(logging.NullHandler())


def init_logging():
    """Read the logging configuration from conf/ and initialize."""
    global log

    def _env(key, default):
        if key in os.environ:
            return os.environ[key]
        return default

    # expeting and removing the NullHandler
    assert len(log.handlers) == 1, "log misconfiguration"
    log.removeHandler(log.handlers[0])

    conf_file = _env(ENV_LOG_CONF, ENV.DIR.CONF / "logging.yaml")
    if not Path(conf_file).exists():
        return

    with kpath(conf_file, is_file=True).open(mode="r") as fd:
        conf = yaml.safe_load(fd)

        logfile = conf["handlers"]["logfile"]
        logfile["filename"] = _env(
            ENV_LOG_FILE,
            logfile["filename"].format(ENV=ENV),
        )

        logfile_libs = conf["handlers"]["logfile_libs"]
        logfile_libs["filename"] = _env(
            ENV_LOG_FILE_LIBS,
            logfile["filename"].format(ENV=ENV),
        )

        logging.config.dictConfig(conf)
        logging.captureWarnings(True)

    log.info("logging initialized")


#
# MISC
#
banner = """
 ┌──────────────────────────────┐
 │ IRT2M COMMAND LINE INTERFACE │
 └──────────────────────────────┘
"""
