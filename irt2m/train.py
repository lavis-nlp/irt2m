# -*- coding: utf-8 -*-
"""Train some neural models for irt2."""

from irt2m import data

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run():
    """Run a training."""
    log.info("commence training")
    data.load_tokenizer(Path("data/test/tokenizer"))
