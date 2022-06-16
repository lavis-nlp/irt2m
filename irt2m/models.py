# -*- coding: utf-8 -*-
"""Different baseline models for IRT2."""

import enum
import logging
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property
from itertools import zip_longest

import pykeen.datasets
import pykeen.evaluation
import pykeen.models
import pytorch_lightning as pl
import torch
import transformers as tf
import yaml
from irt2.dataset import IRT2
from irt2.types import MID
from ktz.filesystem import path as kpath
from torch import nn

import irt2m
from irt2m import data

log = logging.getLogger(__name__)


# --


OPTIMIZER = {
    "adam": torch.optim.Adam,
}


# --


# not using abc to avoid multi-inheritance
class Base(pl.LightningModule):
    """Base model with common functionality."""

    config: data.Config
    encoder: tf.BertModel
    tokenizer: tf.BertTokenizer

    def __init__(
        self,
        config: data.Config,
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

    config: data.Config
    mid2idx: dict[MID, int]

    model: pykeen.models.ERModel
    dataset: pykeen.datasets.Dataset

    @cached_property
    def idx2mid(self) -> dict[int, MID]:
        return {v: k for k, v in self.mid2idx.items()}

    @property
    def cw_entity_embedding(self) -> nn.Embedding:
        return self.model.entity_representations[0]

    @property
    def relation_embedding(self) -> nn.Embedding:
        return self.model.relation_representations[0]

    @cached_property
    def embedding_dim(self) -> int:
        # this returns 1000 for 500 dimensional complex embeddings
        return self.cw_entity_embedding.embedding_dim

    @cached_property
    def closed_world_idxs(self) -> torch.LongTensor:
        cw = set(range(self.dataset.num_entities)) - set(self.idx2mid)
        return torch.Tensor(sorted(cw)).to(dtype=torch.long)

    @cached_property
    def open_world_idxs(self) -> torch.LongTensor:
        return torch.Tensor(sorted(self.idx2mid)).to(dtype=torch.long)

    def __init__(self, irt2: IRT2, config):
        self.config = config
        path = kpath(config["kgc"], is_dir=True)

        with kpath(path / "config.yaml", is_file=True).open(mode="r") as fd:
            self.config = yaml.safe_load(fd)

        self.model = torch.load(
            kpath(
                path / "pipeline" / "trained_model.pkl",
                is_file=True,
            )
        )

        self.mid2idx, self.dataset = data.create_ow_pykeen_dataset(irt2)


class Evaluation(enum.Enum):

    # original closed world
    kgc_train = "kgc/train"

    # projected closed world
    kgc_transductive = "kgc/transductive"

    # projected open world (validation split)
    kgc_inductive = "kgc/inductive"

    # projected open world (test split)
    kgc_test = "kgc/test"


class Projector(Base):

    # see __init__
    kgc: KGC
    irt2: IRT2
    config: data.Config
    evaluations: set[Evaluation]

    # see _init_projections
    targets: torch.Tensor
    projections: torch.Tensor
    projections_counts: torch.Tensor

    # We use manual optimization (gradient accumulation for multi-context models)
    # https://pytorch-lightning.readthedocs.io/en/1.5.10/common/optimizers.html
    automatic_optimization = False

    @property
    def debug(self) -> bool:
        return self.config["trainer"]["fast_dev_run"]

    # projection management

    def _init_projections(self):
        # init reference embedding

        n = self.kgc.cw_entity_embedding.num_embeddings
        embedding = self.kgc.cw_entity_embedding(torch.arange(n))

        if torch.is_complex(embedding):
            embedding = torch.hstack((embedding.real, embedding.imag))

        assert embedding.dtype == torch.float32

        _, dim = embedding.shape
        total = self.kgc.dataset.num_entities

        # registering buffers to assure that they are (1) not part
        # of any gradient computation and (2) always on the correct device

        self.register_buffer("projections", torch.zeros((total, dim)))
        self.register_buffer("projections_counts", torch.zeros(total))

        # when using a buffer to save reference embeddings, pytorch crashes
        # with "Trying to backward through the graph a second time (or directly
        # access saved tensors after they have already been freed)"... idk
        # self.register_buffer("targets", embedding)
        self.targets = torch.nn.Embedding.from_pretrained(
            embeddings=embedding,
            freeze=True,
        )

        log.info(f"registered projections buffer: {self.projections.shape}")
        log.info(f"registered target buffer: {self.targets}")

    def clear_projections(self):
        """
        Initialize projection buffer

        This needs to be run before every dataloader iteration.
        After text samples have been provided by calling forward(),
        they need to reduced by invoking gather_projections().

        (!) Indexes used for projections are the pykeen entity indexes.
        A mapping of irt2m indexes to pykeen indexes is given by
        the provided pykeen triple factories.

        """
        log.info("! clearing projections buffer")

        self.projections.zero_()
        self.projections_counts.zero_()
        self._gathered_projections = False

    def gather_projections(self):
        log.info("! gathering projections")

        count = int(self.projections_counts.sum().item())
        total = len(self.projections_counts)

        log.info(
            f"averaging {count} projections for"
            f" {total} embeddings ({count/total:2.3f})"
        )

        # calculate averages over all projections
        mask = self.projections_counts != 0
        self.projections[mask] /= self.projections_counts.unsqueeze(1)[mask]

        # TODO assert this reflect context counts of datasets
        self._gathered_projections = True

    def _update_projections(
        self,
        projected: torch.Tensor,
        indexes: list[int],
    ):
        assert len(indexes) == projected.shape[0]

        for v, idx in zip(projected.detach(), indexes):
            self.projections[idx] += v
            self.projections_counts[idx] += 1

    # kgc embedding shenanigans

    def _overwrite_embedding(
        self,
        which: Evaluation,
        new: pykeen.nn.Embedding,
        old: pykeen.nn.Embedding,
    ):

        cw_idxs = self.kgc.closed_world_idxs
        ow_idxs = self.kgc.open_world_idxs

        # train uses the original embeddings
        if which is Evaluation.kgc_train:
            idxs, target = cw_idxs, old._embeddings.weight

        if which is Evaluation.kgc_transductive:
            idxs, target = cw_idxs, self.projections

        if which is Evaluation.kgc_inductive:
            idxs, target = ow_idxs, self.projections

        # TODO (currently disabled)
        # if which is Evaluation.kgc_test:

        log.info(f"replacing {len(idxs)} embeddings")
        new._embeddings.weight[idxs] = target[idxs]

    @contextmanager
    def replace_embeddings(self, which: Evaluation):

        # Unfortunately, the official complex number support of pytorch
        # was already integrated in PyKEEN which (albeit being correct
        # from a swe perspective) complicates things:
        # https://github.com/pykeen/pykeen/blob/fec9fe0f4b160b799ffe09d3acece9ad29367e1d/src/pykeen/nn/representation.py#L343

        # Note: old._embeddings.weight.dtype always returns
        # float even for complex embeddings.

        old = self.kgc.cw_entity_embedding
        dtype = torch.cfloat if old.is_complex else torch.get_default_dtype()
        dims = old.embedding_dim // 2 if old.is_complex else old.embedding_dim

        new = self.kgc.cw_entity_embedding.__class__(
            num_embeddings=self.kgc.dataset.num_entities,
            embedding_dim=dims,
            trainable=False,
            dtype=dtype,
        ).to(device=self.device)

        new._embeddings.weight.zero_()
        self._overwrite_embedding(which, new, old)
        self.kgc.model.entity_representations[0] = new

        # it must be registered as parameter in self.kgc.model somewhere
        # because PyKEENs utils.get_preferred_device gets angry and confused
        # old = old.cpu()

        try:
            yield

        except Exception as exc:
            log.error(f"kgc evaluation error: {exc}")

        finally:
            log.info("restoring original kgc model")
            self.kgc.model.entity_representations[0] = old

    # /kgc embedding shenanigans

    def run_kgc_evaluation(self, which: Evaluation) -> dict:
        print(f"\n\n\nevaluate {which.value} {datetime.now()}\n")
        log.info(f"running >[{which.value}]< evaluation")

        evaluator = pykeen.evaluation.RankBasedEvaluator(filtered=True)
        dataset = self.kgc.dataset

        if which in {
            Evaluation.kgc_transductive,
            Evaluation.kgc_inductive,
            Evaluation.kgc_test,
        }:
            assert self._gathered_projections

        kwargs = {
            Evaluation.kgc_train: dict(
                mapped_triples=dataset.training.mapped_triples,
                additional_filter_triples=[
                    dataset.training.mapped_triples,
                ],
            ),
            Evaluation.kgc_transductive: dict(
                mapped_triples=dataset.training.mapped_triples,
                additional_filter_triples=[
                    dataset.training.mapped_triples,
                ],
            ),
            Evaluation.kgc_inductive: dict(
                mapped_triples=dataset.validation.mapped_triples,
                additional_filter_triples=[
                    dataset.training.mapped_triples,
                ],
            ),
            Evaluation.kgc_test: dict(
                mapped_triples=dataset.testing.mapped_triples,
                additional_filter_triples={
                    dataset.training.mapped_triples,
                    dataset.validation.mapped_triples,
                },
            ),
        }[which]

        if self.debug:
            # choice arbitrary
            log.info("debug mode: reducing mapped triples for scoring")
            kwargs["mapped_triples"] = kwargs["mapped_triples"][:100]

        self.kgc.model.to(device=self.device)
        with self.replace_embeddings(which):
            results = evaluator.evaluate(model=self.kgc.model, **kwargs)

        self.kgc.model.cpu()
        return results

    def _log_kgc_results(self, which: Evaluation, results):
        metrics = {
            "hits@1": results.get_metric("both.realistic.hits_at_1"),
            "hits@5": results.get_metric("both.realistic.hits_at_5"),
            "hits@10": results.get_metric("both.realistic.hits_at_10"),
        }

        self.logger.log_metrics(
            {f"{which.value}/{key}": val for key, val in metrics.items()},
            self.global_step,
        )

        realistic = results.to_dict()["both"]["realistic"]
        log.info(f"{which.value}: >[{realistic['hits_at_10'] * 100:2.3f}]< h@10")
        # TODO unless debug: write out whole result to disk

    # /projection management

    # properties and initialization

    def __init__(self, irt2: IRT2, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.loss = torch.nn.MSELoss()

        # self.embedding = torch.nn.Embedding(1, 1000)
        # self.ff = torch.nn.Linear(1000, 1000)

        self.irt2 = irt2
        self.kgc = KGC(irt2, config)
        self.kgc.model.cpu()

        self.evaluations = {Evaluation(name) for name in config["evaluations"]}
        log.info("Will run evaluations: {{p.value for p in self.evaluations}}")

        self._init_projections()

        print("\nCreated PyKEEN Dataset from IRT2:")
        print(self.kgc.dataset.summary_str())
        print()

    @property
    def subbatch_size(self) -> int:
        return self.config["module"]["train_loader_kwargs"]["subbatch_size"]

    # /properties and initialization

    # lightning callbacks

    def forward(
        self,
        collation: tuple[torch.Tensor, list[data.ProjectorSample]],
    ):
        indexes, samples = collation

        # sub-batch x tokens x text_dims
        encoded = self.encode(indexes)

        # sub-batch x text_dims
        aggregated = self.aggregate(encoded)

        # (unique) keys x text_dims
        reduced, reduced_samples = self.reduce(aggregated, samples)

        # (unique) keys x kge_dims
        projected = self.project(reduced)

        return projected, reduced_samples

    def training_step(
        self,
        collation: tuple[torch.Tensor, list[data.ProjectorSample]],
        batch_idx,
    ):
        # sub-batching and gradient accumulation
        batch, samples = collation

        losses = []
        optimizer = self.optimizers()

        N = len(samples)
        steps = range(0, N, self.subbatch_size)

        subbatches = list(zip_longest(steps, steps[1:], fillvalue=N))
        for j, k in subbatches:

            # unique keys many
            subcollation = batch[j:k], samples[j:k]
            projections, reduced_samples = self.forward(subcollation)

            idxs = [s.key for s in reduced_samples]
            idxs = torch.LongTensor(idxs).to(device=self.device)
            targets = self.targets(idxs)

            loss = self.loss(projections, targets)

            losses.append(loss)
            # self.manual_backward(loss / len(subbatches))
            self.manual_backward(loss)

            # while training, the projections are associated
            # with the respective vertex id: no remapping required
            self._update_projections(
                projections,
                [s.key for s in reduced_samples],
            )

        optimizer.step()
        optimizer.zero_grad()

        loss = torch.stack(losses).mean()

        self.log("training/loss", loss)
        return loss

    def validation_step(
        self,
        collation: tuple[torch.Tensor, list[data.ProjectorSample]],
        batch_idx,
    ):
        projections, reduced_samples = self.forward(collation)

        # while validating, the projections are associated with the
        # respective mention id: remapping to kgc index required
        self._update_projections(
            projections,
            [self.kgc.mid2idx[s.key] for s in reduced_samples],
        )

    def on_fit_start(self):
        log.info("starting to fit")

        train = Evaluation.kgc_train
        if train in self.evaluations:
            self._kgc_train_results = self.run_kgc_evaluation(train)

    def on_fit_end(self):
        log.info("finished fitting")

    def on_train_start(self):
        log.info("starting training")

    def on_train_epoch_start(self):
        log.info(f"starting training >[epoch {self.current_epoch}]<")

        # continuously log the baseline for a nice transductive plot
        train = Evaluation.kgc_train
        if train in self.evaluations:
            self._log_kgc_results(train, self._kgc_train_results)

    def on_train_epoch_end(self):
        log.info(f"training >[epoch {self.current_epoch}]< ended")

    def on_validation_epoch_start(self, *_):
        log.info("validation epoch start")

    def on_validation_epoch_end(self, *_):
        log.info("validation epoch end")

        self.gather_projections()

        evaluations = {Evaluation.kgc_transductive, Evaluation.kgc_inductive}
        for which in evaluations & self.evaluations:
            results = self.run_kgc_evaluation(which)
            self._log_kgc_results(which, results)

    # ---

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

        self.projector = torch.nn.Linear(
            self.encoder.config.hidden_size,
            self.kgc.embedding_dim,
        )

    def aggregate(self, encoded):
        return encoded[:, 0]

    def reduce(self, aggregated, samples):
        return aggregated, samples

    def project(self, reduced: torch.Tensor):
        return self.projector(reduced)


PROJECTORS = {
    "single context affine": SingleAffineProjector,
}
