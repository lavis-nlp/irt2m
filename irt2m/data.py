# -*- coding: utf-8 -*-
"""Load irt2 as a torchdata dataset."""


import irt2m

import logging
from pathlib import Path

import transformers as tf

from typing import Optional


log = logging.getLogger(__name__)


def load_tokenizer(
    path: Path,
    model: Optional[str] = None,
) -> tf.BertTokenizer:
    """
    Load or create a BERT tokenizer with special tokens.

    The model name must match the model to be trained. Special marker
    tokens are added (which are defined in irt2/__init__.py).

    Parameters
    ----------
    path : Path
        Where the tokenizer should be saved to
    model_name : Optional[str]
        One of the models provided by huggingface

    Returns
    -------
    tf.BertTokenizer
        A BERT Tokenizer

    """
    if path.is_dir():
        log.info(f"loading tokenizer from {path}")
        tokenizer = tf.BertTokenizer.from_pretrained(str(path))

    else:
        log.info("creating new tokenizer")
        assert model is not None, "you need to provide a model name"

        cache_dir = str(irt2m.ENV.DIR.CACHE / "lib.transformers")
        tokenizer = tf.BertTokenizer.from_pretrained(
            model,
            cache_dir=cache_dir,
            additional_special_tokens=irt2m.TOKEN.values(),
        )

        log.info("saving tokenizer to cache: {cache_dir}")
        tokenizer.save_pretrained(str(path))

    return tokenizer
