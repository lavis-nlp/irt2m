# -*- coding: utf-8 -*-
"""Load irt2 as a torchdata dataset."""

import gzip
import pickle
import random
import logging
import textwrap
from pathlib import Path
from functools import partial
from collections import deque
from dataclasses import dataclass
from collections import defaultdict

import torch
import pykeen as pk
import pykeen.triples
import transformers as tf
from tqdm import tqdm as _tqdm
from tabulate import tabulate

import torch.utils.data as td
from torch.nn.utils.rnn import pad_sequence

from typing import Literal
from typing import Optional
from typing import Hashable

from irt2.types import VID
from irt2.dataset import IRT2
from irt2.dataset import Context

import irt2m

from ktz.string import args_hash
from ktz.string import encode_line
from ktz.string import decode_line
from ktz.filesystem import path as kpath


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


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
        fac = pk.triples.TriplesFactory.create(mapped_triples=hrt)

        # ratios: a float can be given between 0 and 1.0,
        # non-inclusive. the first set of triples will get this ratio
        # and the second will get the rest; random_state is of type
        # pykeen.typing.torchrandomhint which may be an integer
        training, validation = fac.split(ratios=1 - ratio, random_state=seed)

        self = Self(training=training, validation=validation)

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

Tokens = tuple[int]  # token indexes as maintained by the tokenizer
Kind = Literal["train", "validation", "test"]


MAX_SEQ_LEN = 512


@dataclass
class TokenCache:
    """Simple maintenance class to hold cache references."""

    tokens: Path
    meta: Path

    class Reader:
        """Context manager to read data from cache."""

        meta: dict

        def __iter__(self):
            return self.fd_toks

        def __init__(self, cache):
            self.cache = cache

        def __enter__(self):
            self.fd_toks = gzip.open(str(self.cache.tokens), mode="rb")

            log.info("read meta information from cache")
            with self.cache.meta.open(mode="rb") as fd:
                self.meta = pickle.load(fd)

            return self

        def __exit__(self, *_):
            log.info("closing cache reader")
            self.fd_toks.close()

    class Writer:
        """Context manager to write data to the cache."""

        def __init__(self, cache):
            self.cache = cache

        def __enter__(self):
            self.fd_toks = gzip.open(str(self.cache.tokens), mode="wb")
            return self

        def __exit__(self, *_):
            log.info("closing cache writer")
            self.fd_toks.close()

        def write(self, data: bytes):
            """Write a single data point."""
            self.fd_toks.write(data)

        def write_meta(self, meta: dict):
            """Write meta information."""
            log.info("writing meta information to cache")
            with self.cache.meta.open(mode="wb") as fd:
                pickle.dump(meta, fd)

    @property
    def exists(self) -> bool:
        """Check if the cache has content."""
        return self.tokens.exists() and self.meta.exists()

    @classmethod
    def create(Self, hashv: str):
        """Create a new cache."""
        cached = kpath(irt2m.ENV.DIR.CACHE / "irt2m.data", create=True)

        tokens = cached / f"rings.{hashv}.txt.gz"
        meta = cached / f"rings.{hashv}.pk"

        self = Self(
            tokens=tokens,
            meta=meta,
        )

        log.info(f"created token cache for {hashv} (exists={self.exists})")
        return self


# not using abc to avoid multi-inheritance
class RingDataset(td.Dataset):
    """Offer samples and a batch collator."""

    kind: Kind
    model_name: str
    contexts_per_sample: int

    dataset: IRT2
    tokenizer: tf.BertTokenizer

    rings: dict[Hashable, deque[Tokens]]
    idxs: tuple[Hashable]  # for __getitem__
    meta: dict

    def __len__(self):
        """Get dataset size."""
        return len(self.rings)

    def _init_from_cache(self, cache):
        rings = defaultdict(deque)
        with TokenCache.Reader(cache) as reader:

            total = 0
            for line in tqdm(map(self.decode, reader)):

                key, *tokens = line
                rings[key].append(tuple(tokens))
                total += 1

            meta = reader.meta
            assert meta["total"] == total

        return dict(rings), meta

    def _init_from_contexts(self, cache, contexts):
        meta = {"total": 0, "maxlen": 0, "discarded": 0, "pruned": 0}
        rings = defaultdict(deque)

        with TokenCache.Writer(cache) as writer:

            for context in tqdm(contexts):
                tokenized = tuple(self.tokenizer([context.data])["input_ids"][0])

                if MAX_SEQ_LEN < len(tokenized):
                    meta["discarded"] += 1
                    continue

                key = self.create_key(rings, context)
                rings[key].append(tokenized)

                encoded = self.encode(key, tokenized)
                writer.write(encoded)

                meta["maxlen"] = (
                    meta["maxlen"]
                    if len(tokenized) < meta["maxlen"]
                    else len(tokenized)
                )

                meta["total"] += 1

            writer.write_meta(meta)
        return rings, meta

    def _shuffle_and_prune_rings(self, rings, meta):
        n = self.max_contexts_per_sample

        log.info(f"shuffle and prune rings with {self.seed=}")
        random.seed(self.seed)

        total, pruned = 0, 0

        for key in rings:
            ring = sorted(rings[key])
            random.shuffle(ring)

            if n is not None:
                pruned += len(ring) - n
                ring = ring[:n]

            rings[key] = deque(ring)
            total += len(ring)

        meta["total"] = total
        meta["pruned"] = pruned

        return rings, meta

    def _write_pruned_cache(self, cache, rings, meta):
        log.info("write pruned context cache")
        with TokenCache.Writer(cache) as writer:
            gen = ((key, toks) for key, ring in rings.items() for toks in ring)

            for key, tokenized in gen:
                encoded = self.encode(key, tokenized)
                writer.write(encoded)

            writer.write_meta(meta)

    def _cached_init(self, Contexts):
        # (1) load pruned contexts if cached
        # (2) unless (1): load cached tokenized contexts
        # (3) unless (2): tokenize and prune contexts, save to cache
        #     - save tokenized contexts to cache
        #     - if pruned: save pruned contexts to cache

        if self.max_contexts_per_sample:
            log.info(f"requiring pruned contexts (max: {self.max_contexts_per_sample})")
            hashv = args_hash(
                self.dataset.config,
                self.kind,
                self.max_contexts_per_sample,
            )

            pruned_cache = TokenCache.create(hashv)
            if pruned_cache.exists:
                log.info("cache hit! loading pruned contexts")
                return self._init_from_cache(pruned_cache)

        log.info("requiring all contexts")
        hashv = args_hash(self.dataset.config, self.kind)
        cache = TokenCache.create(hashv)

        # either retrieve from cache
        if cache.exists:
            log.info("cache hit! loading contexts")
            rings, meta = self._init_from_cache(cache)

        # or run tokenization
        else:
            with Contexts() as contexts:
                rings, meta = self._init_from_contexts(cache, contexts)

        # prune (max_contexts_per_sample)
        rings, meta = self._shuffle_and_prune_rings(rings, meta)

        if self.max_contexts_per_sample:
            self._write_pruned_cache(pruned_cache, rings, meta)

        return rings, meta

    def __init__(
        self,
        dataset: IRT2,
        seed: int,
        kind: Kind,
        model_name: str,
        contexts_per_sample: int,
        max_contexts_per_sample: Optional[int] = None,
    ):
        super().__init__()

        self.seed = seed
        self.kind = kind
        self.model_name = model_name
        self.dataset = dataset
        self.contexts_per_sample = contexts_per_sample
        self.max_contexts_per_sample = max_contexts_per_sample

        # tokenize
        self.tokenizer = load_tokenizer(
            path=irt2m.ENV.DIR.DATA / "tokenizer" / model_name,
            model=model_name,
        )

        assert self.tokenizer

        Contexts = dict(
            train=dataset.closed_contexts,
            validation=dataset.open_contexts_val,
            test=dataset.open_contexts_test,
        )[kind]

        rings, meta = self._cached_init(Contexts)

        self.rings = dict(rings)
        self.idxs = tuple(rings)
        self.meta = meta

        log.info(f"loaded text for {len(rings)} vertices")
        log.info(f"contexts: {meta['total']=}, maximum token count: {meta['maxlen']}")

    def str_sample(self, i: int, n: int = 0):
        """Represent a single sample as a string."""
        idxs = self.rings[self.idxs[i]][n]
        toks = self.tokenizer.convert_ids_to_tokens(idxs)

        cutoff = 100

        headline = f"Tokens for {i=} (sentence={n})"
        table = "\n".join(line[:cutoff] for line in tabulate((idxs, toks)).split("\n"))
        sentence = self.tokenizer.decode(idxs)[:cutoff]

        return "\n".join((headline, table, sentence))

    def __getitem__(self, i: int) -> tuple[Hashable, list[Tokens]]:
        """Retrieve a VID and N associated text contexts."""
        vid = self.idxs[i]
        ring = self.rings[vid]

        samples = []
        while len(samples) < self.contexts_per_sample:
            samples.append(ring[0])
            ring.rotate()

        return vid, samples

    @staticmethod
    def collate_fn(batch) -> tuple[torch.Tensor]:
        """Batch samples."""
        breakpoint()  # TODO batch type hint

        # # flatten and pad context sentences
        # ctxs = pad_sequence(
        #     [
        #         torch.Tensor(sentence).to(torch.long)
        #         for _, ctx in batch
        #         for sentence in ctx
        #     ],
        #     batch_first=True,
        # )

        # return (ents, ctxs)

    # ---

    def create_key(self, context: Context) -> Hashable:
        """Define a sample based on a context."""
        raise NotImplementedError()

    def encode(self, key: Hashable, tokenized: Tokens) -> bytes:
        """Encode a sample to a single line."""
        raise NotImplementedError()

    def decode(self, encoded: bytes) -> tuple[Hashable, int, ...]:
        """Decode a single-line sample."""
        raise NotImplementedError()


class OWETrainDataset(RingDataset):
    """OWE projector training dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sanity check

        idx = 0
        vid = self.idxs[idx]

        vid2mid = dict(
            train=self.dataset.closed_mentions,
            validation=self.dataset.open_mentions_val,
            test=self.dataset.open_mentions_test,
        )[self.kind]

        mentions = (f"{self.dataset.mentions[mid]} ({mid=})" for mid in vid2mid[vid])

        print(f"Sample {idx}: {self.dataset.vertices[vid]} ({vid=}):")
        print("  Mentions: " + ", ".join(mentions))
        print(textwrap.indent(self.str_sample(i=idx), "  "))
        print()

    def encode(self, key, tokenized) -> bytes:
        """Encode key and tokens as a flat id list."""
        return encode_line((key,) + tokenized, sep=" ", fn=str)

    def decode(self, encoded: bytes):
        """Decode key, *tokens from single id line."""
        return decode_line(encoded, sep=" ", fn=int)

    def create_key(self, rings, context) -> VID:
        """For OWE-Training, a sample is a closed-world vertex."""
        return self.dataset.mid2vid[context.mid]


if __name__ == "__main__":
    irt2m.init_logging()
    irt2 = IRT2.from_dir("data/irt2/irt2-cde-large")

    ds = OWETrainDataset(
        dataset=irt2,
        seed=irt2.config["create"]["seed"],
        kind="validation",
        model_name="bert-base-cased",
        contexts_per_sample=30,
        max_contexts_per_sample=30,
    )


# --- OPEN WORLD JOINT TRAINING
