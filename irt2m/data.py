# -*- coding: utf-8 -*-
"""Load irt2 as a torchdata dataset."""

import gzip
import logging
import pickle
import random
import textwrap
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import ClassVar, Hashable, Literal, Optional, Union

import pykeen as pk
import pykeen.datasets
import pykeen.models
import pykeen.triples
import pytorch_lightning as pl
import torch
import torch.utils.data as td
import transformers as tf
from irt2.dataset import IRT2, Context
from irt2.types import MID, VID
from ktz.filesystem import path as kpath
from ktz.string import args_hash, decode_line, encode_line
from tabulate import tabulate
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm as _tqdm

import irt2m

TERM_WIDTH = 120

log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=TERM_WIDTH)

Config = dict


def dotrslv(dic, keychain, skiplast: Optional[int] = 0):
    "foo.bar.baz -> dic['foo']['bar']['baz']"

    try:
        for key in keychain.split(".")[:-skiplast]:
            dic = dic[key]
        return dic
    except KeyError:
        return None


# TODO use pykeen.datasets.EagerDataset
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
        cls,
        irt2: IRT2,
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
        triples = sorted(irt2.closed_triples)
        random.shuffle(triples)

        hrt = torch.Tensor(triples).to(dtype=torch.long)[:, (0, 2, 1)]
        fac = pk.triples.TriplesFactory.create(mapped_triples=hrt)

        # ratios: a float can be given between 0 and 1.0,
        # non-inclusive. the first set of triples will get this ratio
        # and the second will get the rest; random_state is of type
        # pykeen.typing.torchrandomhint which may be an integer
        training, validation = fac.split(ratios=1 - ratio, random_state=seed)

        self = cls(training=training, validation=validation)

        log.info(
            f"created PyKEEN with training={self.training.num_triples} "
            f"and validation={self.validation.num_triples}"
        )

        return self

    def to_path_binary(self, path: Path):
        log.info(f"write datasets to {path}")
        self.training.to_path_binary((path / "training").resolve())
        self.validation.to_path_binary((path / "validation").resolve())


# --- OPEN WORLD TRAINING


# index internally used to maintain
# a global, gapless numbering of entitites
# to be scored by the PyKEEN triple scorers
IDX = int


class KGC:
    """
    Wraps a PyKEEN Dataset and handles id-mapping
    """

    name: str
    mid2idx: dict[MID, IDX]
    dataset: pykeen.datasets.Dataset

    @cached_property
    def idx2mid(self) -> dict[IDX, MID]:
        return {v: k for k, v in self.mid2idx.items()}

    @cached_property
    def idx2str(self) -> dict[IDX, str]:
        return {v: k for k, v in self.dataset.entity_to_id.items()}

    @cached_property
    def num_embeddings(self):
        return max(self.dataset.entity_to_id.values()) + 1

    @cached_property
    def closed_world_idxs(self) -> torch.LongTensor:
        # simply use the available vids
        # information of dataset.training.entity_ids etc is not usable
        # because all share the whole entity set

        ht = self.dataset.training.mapped_triples[:, (0, 2)]
        cw = sorted(set(ht.ravel().tolist()))

        return torch.LongTensor(sorted(cw))

    @cached_property
    def open_world_idxs(self) -> torch.LongTensor:
        return torch.LongTensor(sorted(self.idx2mid))

    def _open_world_idxs(
        self,
        mapped_triples,
        filtered: set = None,
    ) -> torch.LongTensor:
        ow = set(self.open_world_idxs.tolist())
        idxs = set(mapped_triples[:, (0, 2)].ravel().tolist())

        idxs = (idxs & ow) - (filtered or set())
        return torch.LongTensor(sorted(idxs))

    @cached_property
    def open_world_idxs_val(self) -> torch.LongTensor:
        mapped_triples = self.dataset.validation.mapped_triples
        return self._open_world_idxs(mapped_triples)

    @cached_property
    def open_world_idxs_test(self) -> torch.LongTensor:
        mapped_triples = self.dataset.testing.mapped_triples
        return self._open_world_idxs(
            mapped_triples,
            filtered=set(self.open_world_idxs_val.tolist()),
        )

    # --

    @cached_property
    def description(self) -> str:
        header = "KGC Dataset:\n"

        counts = [
            "maintained indexes:",
            f"     closed world (train): {len(self.closed_world_idxs)}",
            f"       open world (total): {len(self.open_world_idxs)}",
            f"  open world (validation): {len(self.open_world_idxs_val)}",
            f"        open world (test): {len(self.open_world_idxs_test)}",
        ]
        counts = "\n".join(counts)

        buf = [
            self.dataset.summary_str(),
            counts,
        ]

        return header + textwrap.indent("\n".join(buf), prefix="  ")

    def _create_from_irt2(self, irt2):
        # The KGC models are trained based on a TriplesFactory
        # which only has mapped_triples provided (see data.PyKEEN.from_irt2).
        # The way PyKEEN handles this case is quite simple:
        #  - num_entities: maximum vid + 1
        #  - num_relations: maximum rid + 1
        #  - entity_ids: set(vids)
        #  - relation_ids: set(rids)
        #
        # This means we can safely create a new TriplesFactory with
        # all known triples as no internal id-mapping takes place.
        # However, for open-world predictions, we change to MID's
        # and need to create a mapping for that.

        # (1) create mid -> index mapping

        val_mids = set.union(*irt2.open_mentions_val.values())
        test_mids = set.union(*irt2.open_mentions_test.values())

        # test leakage
        assert not val_mids & test_mids
        mids = val_mids | test_mids

        closed_vids = {v for h, t, r in irt2.closed_triples for v in (h, t)}
        open_vids = set(irt2.vertices) - closed_vids

        offset = max(closed_vids) + 1

        mid2idx = {mid: i + offset for i, mid in enumerate(mids)}

        # test partition
        assert not set(closed_vids) & set(mid2idx.values())

        # (2) map tasks to triples

        closed_vids = {v for h, t, r in irt2.closed_triples for v in (h, t)}

        def triples(col: dict):
            # removing ow-ow triples
            yield from (
                (mid, rid, vid)
                for (mid, rid), vids in col.items()
                for vid in vids
                if vid in closed_vids
            )

        def build_tripleset(head_task, tail_task):
            heads = set(
                (mid2idx[mid], rid, vid)
                for mid, rid, vid in triples(head_task)
                if vid not in open_vids
            )

            tails = set(
                (vid, rid, mid2idx[mid])
                for mid, rid, vid in triples(tail_task)
                if vid not in open_vids
            )

            return heads | tails

        # htr -> hrt as pykeen requires
        train_triples = {(h, r, t) for h, t, r in irt2.closed_triples}

        valid_triples = build_tripleset(
            head_task=irt2.open_kgc_val_heads,
            tail_task=irt2.open_kgc_val_tails,
        )

        test_triples = build_tripleset(
            head_task=irt2.open_kgc_test_heads,
            tail_task=irt2.open_kgc_test_tails,
        )

        # (3) build pykeen datasets

        # NOTE: TriplesFactory.with_labels *silently* discards
        # this information!!!
        # kwargs = dict(
        #     num_entities=max(mid2idx.values()),
        #     num_relations=len(irt2.relations),
        # )

        e2idx = {f"{irt2.vertices[vid]} ({vid=})": vid for vid in closed_vids}
        e2idx |= {f"{irt2.mentions[mid]} ({mid=})": idx for mid, idx in mid2idx.items()}

        r2idx = {name: rid for rid, name in irt2.relations.items()}

        def factory(triples):
            # create a CoreTriplesFactory
            mapped = torch.LongTensor(list(triples))
            fac = pykeen.triples.TriplesFactory.create(mapped_triples=mapped)

            # and add labels (this also checks consistency)!
            labeled = fac.with_labels(entity_to_id=e2idx, relation_to_id=r2idx)

            return labeled

        dataset = pykeen.datasets.EagerDataset(
            training=factory(train_triples),
            validation=factory(valid_triples),
            testing=factory(test_triples),
        )

        self.mid2idx = mid2idx
        self.dataset = dataset

    def __init__(self, irt2: IRT2):
        super().__init__()
        self._create_from_irt2(irt2)

    # index mapping: convert between indexes and vid, mid

    def idx2key(
        self,
        idx: IDX,
    ) -> tuple[Literal["mid", "vid"], Union[VID, MID]]:
        # if it can be found in the KGC.idx2mid mapping
        # it was a former MID and a VID otherwise

        try:
            mid = self.idx2mid[idx]
            return "mid", mid
        except KeyError:
            return "vid", idx

    def key2idx(
        self,
        kind: Literal["mid", "vid"],
        key: Union[VID, MID],
    ) -> IDX:
        assert kind in {"mid", "vid"}

        if kind == "mid":
            return self.mid2idx[key]
        if kind == "vid":
            return key


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
    model : Optional[str]
        One of the models provided by huggingface

    Returns
    -------
    tf.BertTokenizer
        A BERT Tokenizer

    """
    if path.is_dir():
        log.info(f"loading tokenizer from {path}")
        tokenizer = tf.BertTokenizerFast.from_pretrained(str(path))

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


Indexes = tuple[int]  # token indexes as maintained by the tokenizer
Tokens = tuple[str]  # translated token indexes
Kind = Literal["train", "valid", "test"]


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

            log.debug("read meta information from cache")
            with self.cache.meta.open(mode="rb") as fd:
                self.meta = pickle.load(fd)

            return self

        def __exit__(self, *_):
            log.debug("closing cache reader")
            self.fd_toks.close()

    class Writer:
        """Context manager to write data to the cache."""

        def __init__(self, cache):
            self.cache = cache
            self.fd_toks = None  # set in __enter__

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
    def create(cls, hashv: str):
        """Create a new cache."""
        cached = kpath(irt2m.ENV.DIR.CACHE / "irt2m.data", create=True)

        tokens = cached / f"rings.{hashv}.txt.gz"
        meta = cached / f"rings.{hashv}.pk"

        self = cls(tokens=tokens, meta=meta)

        log.info(f"created token cache for {hashv} (exists={self.exists})")
        return self


@dataclass(frozen=True)
class ProjectorSample:

    key: Hashable
    keyname: str

    indexes: tuple[Indexes]

    tokenizer: ClassVar[tf.BertTokenizer]

    def __str__(self) -> str:
        return f"Sample {self.keyname}={self.key}: {len(self.indexes)} contexts"

    @property
    def description(self):
        """Represent a single sample as a string."""
        max_texts = 3
        headline = f"Sample {self.keyname}={self.key}:"

        rows = []
        for indexes, tokens in zip(self.indexes, self.tokens):
            row = [cell for tup in zip(indexes, tokens) for cell in tup]
            rows.append(row)

        table = tabulate(rows[:max_texts])
        table = "\n".join(line[:TERM_WIDTH] for line in table.split("\n"))

        texts = "\n".join(text[:TERM_WIDTH] for text in self.texts[:max_texts])
        return "\n".join((headline, table, texts))

    @property
    def texts(self) -> tuple[str]:
        convert = self.tokenizer.decode
        return [convert(idxs) for idxs in self.indexes]

    @property
    def tokens(self) -> tuple[Tokens]:
        convert = self.tokenizer.convert_ids_to_tokens
        return [convert(idxs) for idxs in self.indexes]


#   Dataset Hierarchie
#   ------------------
#
#   - ABC:RingDataset
#     [handles loading and caching of tokenized contexts]
#
#     - VertexRingDataset [key=VID]
#     - MentionRingDataset [key=MID]
#
#     - ABC:FlatDataset
#       [flattens data to iterate all provided contexts in an epoch]
#       - VertexFlatDataset [key=VID]
#       - MentionFlatDataset [key=MID]
#
#  RingDataset is the base class because it makes the pruning step
#  canonical for every other implementation.


class RingDataset(td.Dataset):
    """Offer samples and a batch collator."""

    kind: Kind  # train, valid, test
    contexts_per_sample: int

    irt2: IRT2
    tokenizer: tf.BertTokenizer

    rings: dict[Hashable, deque[Indexes]]
    idxs: tuple[Hashable]  # for __getitem__
    stats: dict

    def _tqdm(self, *args, **kwargs):
        print()
        yield from tqdm(*args, unit=" contexts", **kwargs)
        print()

    def _init_from_cache(self, cache):
        rings = defaultdict(deque)
        with TokenCache.Reader(cache) as reader:

            stats = reader.meta
            total = 0

            gen = self._tqdm(
                map(self.decode, reader),
                desc="load from cache: ",
                total=stats["total"],
            )

            for line in gen:

                key, mrg0, mrg1, *tokens = line
                rings[key].append(((mrg0, mrg1), tuple(tokens)))
                total += 1

            assert stats["total"] == total

        return dict(rings), stats

    def _init_from_contexts(self, cache, contexts):
        stats = Counter()

        def _mention_range(context, tokenized):
            # cannot use context.data.index(context.mention) to access
            # the tokenized.char_to_token functionality because
            # islam matches islamic and querying with "islam "
            # also does not work if theres punctuation etc.

            mention = tuple(self.tokenizer(context.mention)["input_ids"][1:-1])
            tokens = tokenized["input_ids"]

            def different(t):
                return t[0] != t[1]

            # so we need to match a subsequence in a sequence
            for start in range(len(tokens) - 1):
                zipped = zip(tokens[start:], mention)
                if not any(map(different, zipped)):
                    return start, start + len(mention)

        def _check(context, text_tokens, mention_range):
            assert mention_range
            mention = text_tokens[slice(*mention_range)]

            tokens = self.tokenizer.convert_ids_to_tokens(mention)
            recons = self.tokenizer.convert_tokens_to_string(tokens)

            # e.g. a mention might have "look like-this" which leads
            # to "look like - this" so we need to tokenize and reassamle
            # the context mention also
            ref = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer(context.mention)["input_ids"][1:-1],
                )
            )

            assert recons == ref, f"{recons} != {ref}"

        rings = defaultdict(deque)
        with TokenCache.Writer(cache) as writer:

            gen = self._tqdm(contexts, desc="tokenizing: ")

            for context in gen:
                tokenized = self.tokenizer(context.data)

                text_tokens = tuple(tokenized["input_ids"])
                mention_range = _mention_range(context, tokenized)

                if mention_range is None:
                    stats["no mention"] += 1
                    continue

                # paranoid
                _check(context, text_tokens, mention_range)

                if MAX_SEQ_LEN < len(text_tokens):
                    stats["discarded"] += 1
                    continue

                key = self.create_key(context)
                rings[key].append((mention_range, text_tokens))

                encoded = self.encode(key, mention_range, text_tokens)
                writer.write(encoded)

                stats["maxlen"] = (
                    stats["maxlen"]
                    if len(text_tokens) < stats["maxlen"]
                    else len(text_tokens)
                )

                stats["total"] += 1

            writer.write_meta(stats)
        return rings, stats

    def _shuffle_and_prune_rings(self, rings, stats):
        n = self.max_contexts_per_sample

        log.info(f"shuffle and prune rings with {self.seed=}")
        random.seed(self.seed)

        total, pruned = 0, 0

        for key in rings:
            ring = sorted(rings[key], key=lambda t: t[1])
            random.shuffle(ring)

            if n is not None:
                ring = ring[:n]
                pruned += len(rings[key]) - len(ring)

            rings[key] = deque(ring)
            total += len(ring)

        stats["total"] = total
        stats["pruned"] = pruned

        return rings, stats

    def _write_pruned_cache(self, cache, rings, stats):
        log.info("write pruned context cache")
        with TokenCache.Writer(cache) as writer:
            gen = ((key, data) for key, ring in rings.items() for data in ring)

            for key, (mention_range, tokens) in gen:
                encoded = self.encode(key, mention_range, tokens)
                writer.write(encoded)

            writer.write_meta(stats)

    def _cached_init(self, Contexts):
        # (1) load pruned contexts if cached
        # (2) unless (1): load cached tokenized contexts
        # (3) unless (2): tokenize and prune contexts, save to cache
        #     - save tokenized contexts to cache
        #     - if pruned: save pruned contexts to cache

        hashargs = (
            self.seed,
            self.kind,
            self.irt2.config,
            self.keyname,
        )

        threshold = self.max_contexts_per_sample
        if threshold:
            log.info(f"requiring pruned contexts (max: {threshold})")
            hashv = args_hash(threshold, *hashargs)

            pruned_cache = TokenCache.create(hashv)
            if pruned_cache.exists:
                log.info("cache hit! loading pruned contexts")
                return self._init_from_cache(pruned_cache)

        log.info("requiring all contexts")
        hashv = args_hash(*hashargs)

        # either retrieve from cache
        cache = TokenCache.create(hashv)
        if cache.exists:
            log.info("cache hit! loading contexts")
            rings, stats = self._init_from_cache(cache)

        # or run tokenization
        else:
            with Contexts() as contexts:
                rings, stats = self._init_from_contexts(cache, contexts)

        # prune (max_contexts_per_sample)
        rings, stats = self._shuffle_and_prune_rings(rings, stats)

        if threshold:
            self._write_pruned_cache(pruned_cache, rings, stats)

        return rings, stats

    def __init__(
        self,
        irt2: IRT2,
        tokenizer: tf.BertTokenizer,
        config: Config,
        kind: Kind,
        seed: int = None,
        contexts_per_sample: int = None,
        max_contexts_per_sample: Optional[int] = None,
        keep: Optional[set[int]] = None,
        masking: Optional[bool] = False,
    ):
        assert seed is not None, "fixed seed is required"

        super().__init__()
        log.info(f"initializing >[{kind}]< ringbuffer dataset ({seed=})")
        if keep:
            log.info(f"got a keeplist with {len(keep)} keys")

        self.seed = seed
        self.kind = kind
        self.irt2 = irt2
        self.contexts_per_sample = contexts_per_sample
        self.max_contexts_per_sample = max_contexts_per_sample

        # masking is applied in __getitem__ to share token caches
        self.masking = masking
        log.info(f"contexts will be >[masked: {masking}]<")

        # tokenization

        self.tokenizer = tokenizer
        assert self.tokenizer
        # makes decoding available for sample objects
        ProjectorSample.tokenizer = self.tokenizer

        Contexts = dict(
            train=irt2.closed_contexts,
            valid=irt2.open_contexts_val,
            test=irt2.open_contexts_test,
        )[kind]

        _rings, stats = self._cached_init(Contexts)

        # removing filtered samples
        rings = {k: v for k, v in _rings.items() if not keep or k in keep}
        log.info(f"removed {len(_rings) - len(rings)} keys due to filterlist")

        # register data

        self.rings = dict(rings)
        self.keys = tuple(rings)
        self.stats = stats

        log.info(f"loaded text for {len(rings)} keys")
        log.info(" ".join(": ".join(map(str, tup)) for tup in stats.items()))

    def __len__(self):
        """Get dataset size."""
        return len(self.rings)

    def _mask(self, idxs):
        (start, end), idxs = idxs

        if not self.masking:
            return idxs

        mask = self.tokenizer.vocab[self.tokenizer.mask_token]
        return idxs[:start] + (mask,) + idxs[end:]

    def __getitem__(self, i: int) -> ProjectorSample:
        """Retrieve a VID and N associated text contexts."""
        key = self.keys[i]
        ring = self.rings[key]

        idxlis = []
        while len(idxlis) < self.contexts_per_sample:
            idxs = self._mask(ring[0])
            idxlis.append(idxs)
            ring.rotate()

        sample = ProjectorSample(
            key=key,
            keyname=self.keyname,
            indexes=tuple(idxlis),
        )

        return sample

    @staticmethod
    def collate_fn(
        batch: list[ProjectorSample],
    ) -> tuple[torch.Tensor, list[ProjectorSample]]:
        """Batch samples."""

        # flatten and pad context sentences
        gen = ((idxs, sample) for sample in batch for idxs in sample.indexes)
        indexes, samples = zip(*(gen))
        padded = pad_sequence(
            [torch.Tensor(idxs).to(torch.long) for idxs in indexes],
            batch_first=True,
        )

        return padded, samples

    def encode(
        self,
        key: Hashable,
        mention_range: tuple[IDX],
        tokenized: Indexes,
    ) -> bytes:
        """Encode a sample to a single line."""
        return encode_line((key,) + mention_range + tokenized, sep=" ", fn=str)

    def decode(self, encoded: bytes) -> tuple[Hashable, int, ...]:
        """Decode a single-line sample."""
        return decode_line(encoded, sep=" ", fn=int)

    # --- interface

    def create_key(self, context: Context) -> Hashable:
        """Define a sample based on a context."""
        raise NotImplementedError()


class FlatDataset(RingDataset):
    def __len__(self):
        return len(self._flat)

    def __getitem__(self, i: int):
        key, idxs = self._flat[i]
        idxs = self._mask(idxs)

        return ProjectorSample(
            key=key,
            keyname=self.keyname,
            indexes=(idxs,),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        log.info(f"flattening rings for >[{self.kind}]<")
        self._flat = []

        for key, ring in self.rings.items():
            for indexes in ring:
                self._flat.append((key, (tuple(indexes))))


class VertexDataset:

    keyname = "vid"

    def create_key(self, context) -> VID:
        """A sample is a closed-world vertex."""
        return self.irt2.mid2vid[context.mid]


class MentionDataset:

    keyname = "mid"

    def create_key(self, context) -> VID:
        """A sample is a closed-world vertex."""
        return context.mid


# ---

# Multi-inheritance in python is not multi-inheritance
# per-se but a MRO mixin of classes. It is important
# to write the vertex/mention-dataset first so that
# per MRO create_key can be used when building a ring-
# or flat-dataset.


class VertexRingDataset(VertexDataset, RingDataset):
    """
    Projector training dataset.

    Each sample is a vertex with associated text contexts
    of all its mentions.
    """

    def __init__(self, *args, **kwargs):
        log.info("create >[vertex ringbuffer]< dataset")
        super().__init__(*args, **kwargs)
        log.info(f"created vertex ringbuffer dataset with {len(self)} samples")

    @property
    def description(self) -> str:
        idx = random.randint(0, len(self))
        sample = self[idx]
        vid = sample.key

        vid2mid = dict(
            train=self.irt2.closed_mentions,
            valid=self.irt2.open_mentions_val,
            test=self.irt2.open_mentions_test,
        )[self.kind]

        max_mentions = 5

        mentions = ", ".join(
            f"{self.irt2.mentions[mid]} ({mid=})"
            for mid in list(vid2mid[vid])[:max_mentions]
        )

        header = textwrap.dedent(
            f"""
            PROJECTOR VERTEX DATASET

            Sample {idx}: {self.irt2.vertices[vid]} ({vid=}):
              Mentions ({max_mentions}/{len(vid2mid[vid])}:
              {mentions}

            """
        )

        body = textwrap.indent(sample.description, prefix="  ")
        return header + body


class MentionRingDataset(MentionDataset, RingDataset):
    """
    Each sample is a mention with associated text contexts.
    """

    def __init__(self, *args, **kwargs):
        log.info("create >[mention ringbuffer]< dataset")
        super().__init__(*args, **kwargs)
        log.info(f"created mention ringbuffer dataset with {len(self)} samples")

    @property
    def description(self) -> str:
        idx = random.randint(0, len(self))
        sample = self[idx]
        mid = sample.key

        header = textwrap.dedent(
            f"""
            MENTION RING DATASET
            Example Sample {idx}: {self.irt2.mentions[mid]} ({mid=}):
            """
        )

        body = textwrap.indent(sample.description, prefix="  ")
        return header + body


class VertexFlatDataset(VertexDataset, FlatDataset):
    def __init__(self, *args, **kwargs):
        log.info("create >[flat vertex]< dataset")
        super().__init__(*args, **kwargs)
        log.info(f"initialized vertex flat dataset with {len(self)} samples")

    @property
    def description(self) -> str:
        idx = random.randint(0, len(self))
        sample = self[idx]
        vid = sample.key

        vid2mid = dict(
            train=self.irt2.closed_mentions,
            valid=self.irt2.open_mentions_val,
            test=self.irt2.open_mentions_test,
        )[self.kind]

        max_mentions = 5

        mentions = ", ".join(
            f"{self.irt2.mentions[mid]} ({mid=})"
            for mid in list(vid2mid[vid])[:max_mentions]
        )

        header = textwrap.dedent(
            f"""
            VERTEX FLAT DATASET ({len(self)} samples)
            Sample {idx}: {self.irt2.vertices[vid]} ({vid=}):
              Mentions ({max_mentions}/{len(vid2mid[vid])}:
              {mentions}

            """
        )

        body = textwrap.indent(sample.description, prefix="  ")
        return header + body


class MentionFlatDataset(MentionDataset, FlatDataset):
    """
    Iterates all texts in an epoch: used for validation/test.
    """

    def __init__(self, *args, **kwargs):
        log.info("create >[flat mention]< dataset")
        super().__init__(*args, **kwargs)
        log.info(f"initialized mention flat dataset with {len(self)} samples")

    @property
    def description(self) -> str:
        idx = random.randint(0, len(self))
        sample = self[idx]
        mid = sample.key

        header = textwrap.dedent(
            f"""
            MENTION FLAT DATASET ({len(self)} samples)
            Example Sample {idx}: {self.irt2.mentions[mid]} ({mid=}):
            """
        )

        body = textwrap.indent(sample.description, prefix="  ")
        return header + body


# triples


@dataclass(frozen=True)
class TripleSample(ProjectorSample):
    pass


class VertexTripleRingbuffer(RingDataset):
    def __init__(self, *args, **kwargs):
        log.info("create >[vertex triple ringbuffer]< dataset")
        super().__init__(*args, **kwargs)
        log.info("created triple dataset with {len(self)} samples")


PROJECTOR_DATASETS = {
    "vertex ringbuffer": VertexRingDataset,
    "vertex triple ringbuffer": VertexTripleRingbuffer,
    "vertex flat": VertexFlatDataset,
    "mention ringbuffer": MentionRingDataset,
    "mention flat": MentionFlatDataset,
}


class ProjectorTrainLoader(td.DataLoader):

    subbatch_size: int

    def __init__(
        self,
        *args,
        batch_size: int = None,
        subbatch_size: Optional[int] = None,
        **kwargs,
    ):
        assert batch_size
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.subbatch_size = subbatch_size or batch_size


class ProjectorModule(pl.LightningDataModule):

    irt2: IRT2
    config: Config
    tokenizer: tf.BertTokenizer

    def __init__(
        self,
        irt2: IRT2,
        config: Config,
    ):
        super().__init__(self)
        self.irt2 = irt2
        self.config = config

        self.tokenizer = load_tokenizer(
            path=irt2m.ENV.DIR.DATA / "tokenizer" / self.config["encoder"],
            model=self.config["encoder"],
        )

    def setup(self, stage: Optional[callable]):
        log.info("! setup datasets")

        modconf = self.config["module"]

        if hasattr(self, "trainer"):
            kgc = self.trainer.model.kgc

            def _get(idx):
                return kgc.idx2key(idx)[1]

            keep_train = set(map(_get, kgc.closed_world_idxs.tolist()))
            keep_valid = set(map(_get, kgc.open_world_idxs_val.tolist()))
        else:
            keep_train = None
            keep_valid = None

        TrainDataset = PROJECTOR_DATASETS[modconf["train_ds"]]
        self.train_ds = TrainDataset(
            irt2=self.irt2,
            tokenizer=self.tokenizer,
            config=self.config,
            kind="train",
            keep=keep_train,
            **modconf["train_ds_kwargs"],
        )

        print(self.train_ds.description)

        ValidationDataset = PROJECTOR_DATASETS[modconf["valid_ds"]]
        self.valid_ds = ValidationDataset(
            irt2=self.irt2,
            tokenizer=self.tokenizer,
            config=self.config,
            kind="valid",
            keep=keep_valid,
            **modconf["valid_ds_kwargs"],
        )

        print(self.valid_ds.description)

        self.train_loader_kwargs = modconf["train_loader_kwargs"]
        self.valid_loader_kwargs = modconf["valid_loader_kwargs"]

    def train_dataloader(self):
        return ProjectorTrainLoader(
            self.train_ds,
            collate_fn=self.train_ds.collate_fn,
            **self.train_loader_kwargs,
            # sampler=self.sampler,  # TODO
        )

    def val_dataloader(self):
        return td.DataLoader(
            self.valid_ds,
            collate_fn=self.valid_ds.collate_fn,
            **self.valid_loader_kwargs,
        )

    def test_dataloader(self):
        raise NotImplementedError()

    def transfer_batch_to_device(self, collation, device, dataloader_idx):
        batch, samples = collation
        return batch.to(device=device), samples


# --- OPEN WORLD JOINT TRAINING
# TODO
