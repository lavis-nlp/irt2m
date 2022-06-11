# -*- coding: utf-8 -*-
"""Different baseline models for IRT2."""

import logging
from functools import cached_property
from itertools import zip_longest
from pathlib import Path

import pykeen.models
import pytorch_lightning as pl
import torch
import transformers as tf
import yaml
from irt2.dataset import IRT2
from irt2.types import MID
from ktz.filesystem import path as kpath
from pykeen.triples import TriplesFactory
from torch import nn

import irt2m
from irt2m.data import Config, ProjectorSample

log = logging.getLogger(__name__)


# --


OPTIMIZER = {
    "adam": torch.optim.Adam,
}


# --


# not using abc to avoid multi-inheritance
class Base(pl.LightningModule):
    """Base model with common functionality."""

    config: Config
    encoder: tf.BertModel
    tokenizer: tf.BertTokenizer

    def __init__(
        self,
        config: Config,
        tokenizer: tf.BertTokenizer,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        log.info(f"loading encoder model: {config['encoder']}")
        self.encoder = tf.BertModel.from_pretrained(
            config["encoder"],
            cache_dir=irt2m.ENV.DIR.CACHE / "lib.transformers",
        )

        if freeze_encoder:
            log.info("! freezing text encoder")
            self.encoder.requires_grad_(False)

    def configure_optimizers(self):
        optimizer = OPTIMIZER[self.config["optimizer"]](
            self.parameters(), **self.config["optimizer_kwargs"]
        )

        log.info(
            f"initialized {self.config['optimizer']} with"
            f" {self.config['optimizer_kwargs']}"
        )

        return [optimizer]

    def encode(self, batch):
        mask = self.tokenizer.mask_token_id

        attention_mask = (batch > 0) | (batch == mask)
        attention_mask = attention_mask.to(dtype=torch.long)

        encoded = self.encoder(
            input_ids=batch,
            attention_mask=attention_mask,
        )

        return encoded[0]

    def forward(self, batch):
        raise NotImplementedError()


# --- OPEN WORLD PROJECTOR TRAINING


class KGC:
    """
    Maintains a trained PyKEEN model.
    """

    config: Config
    model: pykeen.models.ERModel

    train_triples: TriplesFactory
    valid_triples: TriplesFactory
    test_triples: TriplesFactory

    mid2idx: dict[MID, int]

    @cached_property
    def idx2mid(self) -> dict[int, MID]:
        return {v: k for k, v in self.mid2idx.items()}

    @property
    def entities(self) -> nn.Embedding:
        return self.model.entity_representations

    @property
    def relations(self) -> nn.Embedding:
        return self.model.relation_representations

    def __init__(self, irt2: IRT2, path: Path):
        with kpath(path / "config.yaml", is_file=True) as fd:
            self.config = yaml.safe_load(fd)

        self.model = torch.load(
            kpath(
                path / "pipeline" / "trained_model.pkl",
                is_file=True,
            )
        )

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

        assert not val_mids & test_mids
        mids = val_mids | test_mids

        offset = self.model.num_entities
        self.mid2idx = {mid: i + offset for i, mid in enumerate(mids)}

        # (2) map tasks to triples

        closed_vids = {v for h, t, r in irt2.closed_triples for v in (h, t)}

        def gen_triples(col: dict):
            # removing ow-ow triples
            yield from (
                (mid, rid, vid)
                for (mid, rid), vids in col.items()
                for vid in vids
                if vid in closed_vids
            )

        def build_tripleset(head_task, tail_task):
            # htr -> hrt as per pykeen requirement

            heads = set(
                (self.mid2idx[mid], rid, vid)
                for mid, rid, vid in gen_triples(head_task)
            )

            tails = set(
                (vid, rid, self.mid2idx[mid])
                for mid, rid, vid in gen_triples(tail_task)
            )

            return heads | tails

        valid_triples = build_tripleset(
            head_task=irt2.open_kgc_val_heads,
            tail_task=irt2.open_kgc_val_tails,
        )

        log.info(f"constructed {len(valid_triples)} validation triples")

        test_triples = build_tripleset(
            head_task=irt2.open_kgc_test_heads,
            tail_task=irt2.open_kgc_test_tails,
        )

        log.info(f"constructed {len(test_triples)} validation triples")

        breakpoint()


class Projector(Base):

    # projection management

    def init_projections(self):
        """
        Initialize projection buffer

        This needs to be run before every dataloader iteration.
        After text samples have been provided by calling forward(),
        they need to reduced by invoking gather_projections().

        (!) Indexes used for projections are the pykeen entity indexes.
        A mapping of irt2m indexes to pykeen indexes is given by
        the provided pykeen triple factories.

        We use manual optimization (gradient accumulation for
        multi-context models)
        https://pytorch-lightning.readthedocs.io/en/1.5.10/common/optimizers.html

        """
        log.info("clearing projections buffer")

        self.projections.zero_()
        self.projections_counts.zero_()
        self._gathered_projections = False

    def gather_projections(self):
        count = int(self.projections_counts.sum().item())
        total = len(self.projections_counts)

        log.info(f"averaging {count} projections for {total} embeddings")

        # calculate averages over all projections
        mask = self.projections_counts != 0
        self.projections[mask] /= self.projections_counts.unsqueeze(1)[mask]

        # TODO assert this reflect context counts of datasets
        self._gathered_projections = True

    def _update_projections(
        self,
        projected: torch.Tensor,
        samples: list[ProjectorSample],
    ):
        assert len(samples) == projected.shape[0]
        log.error("_update_projections: FIX IDX")

        for v, s in zip(projected.detach(), samples):

            # sample.key is the VID which needs to be mapped to
            # the internal id used by pykeen

            idx = 0

            self.projections[idx] += v
            self.projections_counts[idx] += 1

    # /projection management

    # properties and initialization

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.automatic_optimization = False
        breakpoint()

    @property
    def subbatch_size(self) -> int:
        return self.config["module"]["train_loader_kwargs"]["subbatch_size"]

    # /properties and initialization

    # lightning callbacks

    def configure_optimizers(self):
        config = self.irtmc.config

        optimizer = OPTIMIZER[config.optimizer](
            self.parameters(), **config.optimizer_args
        )

        scheduler_name = config.scheduler or "constant"
        fn, kwargs = SCHEDULER[scheduler_name]

        last_epoch = self.current_epoch - 1
        args = [config.scheduler_args[k] for k in kwargs] + [last_epoch]
        scheduler = fn(optimizer, *args)

        log.info(f"initialized optimizer with {config.optimizer_args}")
        log.info(f"initialized {scheduler_name} scheduler with {kwargs=}")
        return [optimizer], [scheduler]

    def forward(self, collation: tuple[torch.Tensor, list[ProjectorSample]]):
        indexes, samples = collation

        # sub-batch x tokens x text_dims
        encoded = self.encode(indexes)

        # sub-batch x text_dims
        aggregated = self.aggregate(encoded)

        # (unique) keys x text_dims
        reduced_samples, reduced = self.reduce(aggregated, samples)

        # (unique) keys x kge_dims
        projected = self.project(reduced)

        self._update_projections(projected, reduced_samples)
        return projected, reduced_samples

    def training_step(
        self,
        collation: tuple[torch.Tensor, list[ProjectorSample]],
        batch_idx,
    ):
        # sub-batching and gradient accumulation
        batch, samples = collation

        losses = []
        optimizer = self.optimizers()

        N = len(samples)
        steps = range(0, N, self.subbatch_size)

        optimizer.zero_grad()

        subbatches = list(zip_longest(steps, steps[1:], fillvalue=N))
        for j, k in subbatches:

            # unique keys many
            subcollation = batch[j:k], samples[j:k]
            projections, reduced_samples = self.forward(subcollation)

            targets = self.kge(reduced_samples)
            loss = self.loss(projections, targets)

            losses.append(loss)
            self.manual_backward(loss / len(subbatches))

        optimizer.step()
        # optimizer.zero_grad()

        loss = torch.stack(losses).mean()

        self.log("train_loss_step", loss)
        return loss

    # /lightning callbacks

    # interface

    # batch x tokens x text_dims -> batch x text_dims
    def aggregate(self, encoded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    # batch x textdims -> [unique] keys x text_dims
    def reduce(self, aggregated, samples):
        raise NotImplementedError()

    # [unique] keys x kge dims
    def project(self, reduced: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    # /interface


class SingleAffineProjector(Projector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        input_dims = self.encoder.config.hidden_size
        log.error("FIX PHONY OUTPUT DIMS")
        output_dims = 100

        self.projector = nn.Linear(input_dims, output_dims)
        log.info(f"initialize projector: {input_dims} -> {output_dims}")

    def aggregate(self, encoded):
        return encoded[:, 0]

    def reduce(self, aggregated, samples):
        return aggregated, samples

    def project(self, reduced: torch.Tensor):
        breakpoint()
        return self.projector(reduced)


PROJECTORS = {
    "single context affine": SingleAffineProjector,
}
