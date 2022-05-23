# -*- coding: utf-8 -*-
"""Load irt2 as a torchdata dataset."""

import irt2m
from irt2.dataset import IRT2

import random
import logging
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import torch
import pykeen as pk
import pykeen.triples
import transformers as tf

from typing import Optional


log = logging.getLogger(__name__)


# --- CLOSED WORLD KGC TRAINING


@dataclass
class PyKEEN:
    """Train KGC models with PyKEEN."""

    training: pk.triples.TriplesFactory
    validation: pk.triples.TriplesFactory
    # we don't need a test set

    def __str__(self):
        """Create string representation."""
        return (
            "IRT2M PyKEEN Dataset: "
            f"{self.training.num_triples} training samples "
            f"{self.validation.num_triples} validation samples "
        )

    @classmethod
    def from_irt2(
        Self,
        dataset: IRT2,
        ratio: float,
        seed: int,
    ):
        """Create a PyKEEN dataset for KGC training.

        Parameters
        ----------
        Self : PyKEEN
        dataset : IRT2
            The source IRT2 dataset
        ratio : float
            Split ratio (0.2 means 20% validation triples)
        seed : int
            For reproducibility

        """
        log.info(f"create PyKEEN dataset with {seed=}")

        random.seed(seed)
        triples = sorted(dataset.closed_triples)
        random.shuffle(triples)

        hrt = torch.Tensor(triples).to(dtype=torch.long)[:, (0, 2, 1)]
        threshold = int(hrt.shape[0] * (1 - ratio))

        assert threshold < hrt.shape[0]

        cw_v = set(hrt[:, 0].tolist()) | set(hrt[:, 2].tolist())
        e2id = {dataset.vertices[v]: v for v in cw_v}
        r2id = {v: k for k, v in dataset.relations.items()}

        log.info(f"retaining {len(e2id)} vertices for training")

        Factory = partial(
            pk.triples.TriplesFactory,
            entity_to_id=e2id,
            relation_to_id=r2id,
        )

        self = Self(
            training=Factory(mapped_triples=hrt[:threshold]),
            validation=Factory(mapped_triples=hrt[threshold:]),
        )

        log.info(
            f"created PyKEEN with training={self.training.num_triples} "
            f"and validation={self.validation.num_triples}"
        )

        return self


# --- OPEN WORLD TRAINING


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


# --- OPEN WORLD PROJECTOR TRAINING

# --- OPEN WORLD JOINT TRAINING
