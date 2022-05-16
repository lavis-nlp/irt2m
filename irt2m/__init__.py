# -*- coding: utf-8 -*-

"""IRT2M import-time things."""


import os
import logging
import logging.config
from enum import Enum
from pathlib import Path

import yaml
from ktz.filesystem import path as kpath


_root_path = kpath(__file__).parent.parent


# check whether data directory is overwritten
ENV_DIR_DATA = "IRT2M_DATA"
if ENV_DIR_DATA in os.environ:
    _data_path = kpath(os.environ[ENV_DIR_DATA])
else:
    _data_path = kpath(_root_path / "data", create=True)


class _DIR:

    ROOT: Path = _root_path
    DATA: Path = _data_path
    CONF: Path = kpath(_root_path / "conf", create=True)
    CACHE: Path = kpath(_data_path / "cache", create=True)


class ENV:
    """IRT2 environment."""

    DIR = _DIR


class IRT2Error(Exception):
    """General error."""

    pass


# special tokenizer tokens
class TOKEN(Enum):
    """Special tokens used by the BERT Tokenizer."""

    mask = "[MASK]"
    mention_start = "[MENTION_START]"
    mention_end = "[MENTION_END]"

    @classmethod
    def values(Self):
        """Get a list of token identifier."""
        return [e.value for e in Self]


log = logging.getLogger(__name__)
# if used as library do not log anything
log.addHandler(logging.NullHandler())


def init_logging():
    """Read the logging configuration from conf/ and initialize."""
    # remove the null handler
    global log

    assert len(log.handlers) == 1, "log misconfiguration"
    log.removeHandler(log.handlers[0])

    with (ENV.DIR.CONF / "logging.yaml").open(mode="r") as fd:
        conf = yaml.safe_load(fd)

        # use data dir configuration to set logfile location
        logfile = conf["handlers"]["logfile"]
        logfile["filename"] = (ENV.DIR.DATA / logfile["filename"]).resolve()

        logging.config.dictConfig(conf)

    # i have no idea how to apply the new configuration to the old logger :/
    # so lets just delete the old logger and initialize a new one
    log.info("logging initialized")
