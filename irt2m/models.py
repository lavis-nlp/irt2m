# -*- coding: utf-8 -*-
"""Different baseline models for IRT2."""

import enum
import logging
import sys
from contextlib import contextmanager
from itertools import count, groupby, zip_longest
from pathlib import Path
from typing import Literal

import pykeen.datasets
import pykeen.evaluation
import pykeen.models
import pytorch_lightning as pl
import torch
import transformers as tf
import yaml
from irt2.dataset import IRT2
from ktz.filesystem import path as kpath
from torch import Tensor

import irt2m
from irt2m import data

log = logging.getLogger(__name__)

# suppress the annoying bert-base* load warnings
tf.logging.set_verbosity_error()


# --


OPTIMIZER = {
    "adam": torch.optim.Adam,
}


# --


class Evaluation(enum.Enum):

    # original closed world
    kgc_train = "kgc/train"

    # projected closed world
    kgc_transductive = "kgc/transductive"

    # projected open world (validation split)
    kgc_inductive = "kgc/inductive"

    # projected open world (test split)
    kgc_test = "kgc/test"


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

    # kgc

    def run_kgc_evaluation(self, which: Evaluation) -> dict:
        self.targets = self.targets.to(device=self.device)

        # print(f"\n\n\nevaluate {which.value} {datetime.now()}\n")
        # log.info(f"running >[{which.value}]< evaluation")

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

        model = self.kgc_model.model.to(device=self.device)
        with self.replace_embeddings(which):
            results = evaluator.evaluate(
                model=model,
                tqdm_kwargs=dict(
                    leave=False,
                    ncols=data.TERM_WIDTH,
                    file=sys.stdout,
                ),
                **kwargs,
            )

        # self.kgc_model = self.kgc_model.cpu()
        return results

    def _log_kgc_results(self, which: Evaluation, results):
        metrics = {
            "hits@1": results.get_metric("both.realistic.hits_at_1"),
            "hits@5": results.get_metric("both.realistic.hits_at_5"),
            "hits@10": results.get_metric("both.realistic.hits_at_10"),
        }

        self.log_dict({f"{which.value}/{key}": val for key, val in metrics.items()})

        realistic = results.to_dict()["both"]["realistic"]
        log.info(f"{which.value}: >[{realistic['hits_at_10'] * 100:2.3f}]< h@10")

        if not self.debug:

            fname = (
                f"epoch={self.current_epoch}"
                f"-step={self.global_step}"
                f"_{which.value.split('/')[1]}"
                ".yaml"
            )

            path = kpath(self.config["out"], is_dir=True)
            path = kpath(path / "kgc", create=True)

            with (path / fname).open(mode="w") as fd:
                fd.write(yaml.safe_dump(results.to_dict()))


# --- OPEN WORLD PROJECTOR TRAINING


class KGCModel:
    """
    Maintains a trained PyKEEN 1.8.1 model.

    Some notes regarding embeddings as maintained by PyKEEN:

    Data Model
    ----------

    model = self.kgc_model

    model.entity_representations:
      torch.nn.modules.container.ModuleList

    model.entity_representations[0]:
      pykeen.nn.representation.Embedding

    model.entity_representations[0]._embeddings:
      torch.nn.modules.sparse.Embedding

    model.entity_representations[0]._embeddings.weight:
      torch.nn.parameter.Parameter


    Data Types
    ----------

    Let's see for ComplEx where the complex numbers are:

    idxs = torch.LongTensor([0])
    f = torch.is_complex

    f(model.entity_representations[0]._embeddings.weight)
      False

    f(self.model.entity_representations[0]._embeddings(idxs))
      False

    f(self.model.entity_representations[0](idxs))
      True

    This is because PyKEEN saves data as real-valued (n, 2*d) sized
    real embeddings and in the forward method torch.view_as_complex
    is called (torch.nn.Embedding has no complex number support yet).

    Side note:
      - view_as_complex: N x D x 2 -> N x D
      - view_as_real:    N x D -> N x D x 2
      - PyKEEN seems to encode it as N x 2D where the data is saved
        with stride; for a complex vector [c_1, c_2, ...] c_1 = r_1 + i * i_1
        the view_as_real view would be [[r_1, r_2, ...], [i_1, i_2, ...]]
        and PyKEEN saves it as [r_1, c_1, r_2, c_2, ...]

    See also:
     - https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/nn/representation.py#L353
     - https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/nn/representation.py#L410


    Conclusion
    ----------

    We can simply access the private _embeddings.weight and leave
    the interpretation for PyKEEN when scoring as we are only
    interested in geometric difference (as in norm). And this feature
    -wise difference allows us to to leave the interpretation opaque.
    """

    config: dict
    path: Path
    model: pykeen.models.ERModel

    def __init__(self, config):
        assert len(config["kgc"]) == 1
        self.name, path = list(config["kgc"].items())[0]
        self.path = kpath(path, is_dir=True)

        log.info(f"loading >[{self.name}]< kgc model from {path}")

        with kpath(self.path / "config.yaml", is_file=True).open(mode="r") as fd:
            self.config = yaml.safe_load(fd)

        model = kpath(self.path / "pipeline" / "trained_model.pkl", is_file=True)
        self.model = torch.load(model)


class Projector(Base):

    # see __init__
    irt2: IRT2

    kgc: data.KGC
    kgc_model: KGCModel

    evaluations: set[Evaluation]

    # see _init_projections
    targets: Tensor
    projections: Tensor
    projections_counts: Tensor

    # We use manual optimization (gradient accumulation for multi-context models)
    # https://pytorch-lightning.readthedocs.io/en/1.5.10/common/optimizers.html
    automatic_optimization = False

    @property
    def debug(self) -> bool:
        return self.config["trainer"]["fast_dev_run"]

    # ----------------------------------------
    # projection management

    def _init_projections(self):
        # init reference embedding
        self.targets = self.cw_entity_embedding._embeddings.weight.detach()
        self.targets = self.targets.to(device=self.device)

        # we require real embeddings for norm/euclidean operations
        assert self.targets.dtype == torch.float32

        _, dim = self.targets.shape
        total = self.kgc.num_embeddings
        # total = self.kgc.dataset.num_entities

        # registering buffers to assure that they are (1) not part
        # of any gradient computation and (2) always on the correct device

        self.register_buffer("projections", torch.zeros((total, dim)))
        self.register_buffer("projections_counts", torch.zeros(total))

        # note: when using a buffer to save reference embeddings, pytorch crashes
        # with "Trying to backward through the graph a second time (or directly
        # access saved tensors after they have already been freed)"
        # self.register_buffer("targets", embedding)

        log.info(f"registered projections buffer: {self.projections.shape}")
        log.info(f"registered target tensor: {self.targets.shape}")
        self._gathered_projections = False

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

    def _log_projection_stats(self):
        def _perc(f):
            return f"{round(f.item() * 100)}%"

        def _stats(which, idxs):
            projections = self.projections_counts[idxs]

            mask = projections != 0
            count = mask.sum()
            total = int(projections[mask].sum().item())
            ratio = total / count
            perc = _perc(count / len(idxs))

            log.info(
                f" >[{which} projections]< "
                f" {count}/{len(idxs)} ({perc}) items"
                f" with {total} projections ({ratio=:.2f})"
            )

        _stats("transductive", self.kgc.closed_world_idxs)
        _stats("inductive", self.kgc.open_world_idxs_val)
        _stats("test", self.kgc.open_world_idxs_test)

    def gather_projections(self, force: bool = False):
        if not force and self.global_step == 0:
            log.warning("skipping projection gathering (not trained yet)!")
            self.clear_projections()
            self._gathered_projections = True
            return

        self._log_projection_stats()

        # calculate averages over all projections
        mask = self.projections_counts != 0
        self.projections[mask] /= self.projections_counts.unsqueeze(1)[mask]

        # TODO assert this reflect context counts of datasets
        self._gathered_projections = True

    def update_projections(
        self,
        projected: Tensor,
        samples: list[data.ProjectorSample],
    ):
        assert len(samples) == projected.shape[0]

        idxs = [self.kgc.key2idx(s.keyname, s.key) for s in samples]

        for v, idx in zip(projected.detach(), idxs):
            self.projections[idx] += v
            self.projections_counts[idx] += 1

    def projection_error(self):
        assert self._gathered_projections

        idxs = self.kgc.closed_world_idxs
        mask = self.projections_counts[idxs] != 0

        projections = self.projections[idxs][mask]
        targets = self.targets[idxs][mask]

        error = self.compare(projections, targets)
        return error

    # kgc embedding shenanigans

    def _overwrite_embedding(
        self,
        which: Evaluation,
        new: pykeen.nn.Embedding,
    ):

        cw_idxs = self.kgc.closed_world_idxs
        ow_idxs = self.kgc.open_world_idxs

        # train uses the original embeddings
        # (which also checks whether the targets tensor is correct)
        if which is Evaluation.kgc_train:
            idxs, projections = cw_idxs, self.targets

        if which is Evaluation.kgc_transductive:
            idxs, projections = cw_idxs, self.projections

        if which is Evaluation.kgc_inductive:
            idxs, projections = ow_idxs, self.projections

        if which is Evaluation.kgc_test:
            idxs, projections = ow_idxs, self.projections

        # pykeen directly modifies the _embedding.weight.data
        # https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/nn/representation.py#L395

        log.info(f"replacing {len(idxs)} embeddings for {which.value}")
        new._embeddings.weight[idxs] = projections[idxs]

    @contextmanager
    def replace_embeddings(self, which: Evaluation):
        old = self.cw_entity_embedding

        # https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/models/unimodal/complex.py#L102
        new = self.cw_entity_embedding.__class__(
            max_id=self.kgc.num_embeddings,
            shape=old.shape,
            # https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/models/unimodal/complex.py#L106
            dtype=torch.cfloat if old.is_complex else torch.get_default_dtype(),
            regularizer=old.regularizer,
            trainable=False,
        )

        new = new.to(device=self.device)

        # always provide the closed-world embeddings
        self._overwrite_embedding(Evaluation.kgc_train, new)
        if which is not Evaluation.kgc_train:
            self._overwrite_embedding(which, new)

        self.kgc_model.model.entity_representations[0] = new

        try:
            yield

        except Exception as exc:
            log.error(f"kgc evaluation error: {exc}")

        finally:
            log.info("restoring original kgc model")
            self.kgc_model.model.entity_representations[0] = old

    # /kgc embedding shenanigans
    # /projection management
    # ----------------------------------------
    # kgc model

    @property
    def kgc_embedding_dim(self) -> int:
        # this returns 1000 for 500 dimensional complex embeddings
        return self.cw_entity_embedding.embedding_dim

    @property
    def cw_entity_embedding(self) -> pykeen.nn.representation.Embedding:
        return self.kgc_model.model.entity_representations[0]

    @property
    def relation_embedding(self) -> pykeen.nn.representation.Embedding:
        return self.kgc_model.model.relation_representations[0]

    # /kgc model
    # ----------------------------------------

    def __init__(self, irt2: IRT2, config: dict, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.irt2 = irt2
        self.kgc = data.KGC(irt2)
        self.kgc_model = KGCModel(config)

        self.evaluations = {Evaluation(name) for name in config["evaluations"]}
        log.info(f"will run evaluations: {[p.value for p in self.evaluations]}")

        self._init_projections()

    # ----------------------------------------
    # lightning callbacks and training

    @property
    def subbatch_size(self) -> int:
        return self.trainer.train_dataloader.loaders.subbatch_size

    def _subbatch(self, batch, samples):
        # sub-batching and gradient accumulation
        N = len(samples)
        steps = range(0, N, self.subbatch_size)

        subbatches = list(zip_longest(steps, steps[1:], fillvalue=N))
        for j, k in subbatches:
            yield batch[j:k], samples[j:k]

    # --

    def forward(
        self,
        collation: tuple[Tensor, list[data.ProjectorSample]],
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
        collation: tuple[Tensor, list[data.ProjectorSample]],
        batch_idx,
    ):
        losses = []
        optimizer = self.optimizers()

        batch, samples = collation
        for subcollation in self._subbatch(batch, samples):
            projections, reduced_samples = self.forward(subcollation)

            idxs = [self.kgc.key2idx(s.keyname, s.key) for s in reduced_samples]
            targets = self.targets[idxs]

            loss = self.compare(projections, targets)
            losses.append(loss)

            self.manual_backward(loss)
            self.update_projections(projections, reduced_samples)

        optimizer.step()
        optimizer.zero_grad()

        loss = torch.stack(losses).mean()
        self.log("training/loss", loss)

        return loss

    def validation_step(
        self,
        collation: tuple[Tensor, list[data.ProjectorSample]],
        batch_idx,
    ):
        batch, samples = collation
        projections, reduced_samples = self.forward(collation)
        self.update_projections(projections, reduced_samples)

    def on_fit_start(self):
        print("\n\n")
        print(self.kgc.description)
        print()

        print("\nFITTING\n")
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
        self.log("trainer/global_epoch", self.current_epoch)

    def on_train_epoch_end(self):
        log.info(f"training epoch {self.current_epoch} ended")

    def on_validation_epoch_start(self, *_):
        log.info("validation epoch start")

    def on_validation_epoch_end(self, *_):
        log.info("validation epoch end")

        self.gather_projections()

        # log the average distance between projections
        # and targets (somewhat of a validation loss)

        if not self.global_step == 0:
            error = self.projection_error()
            self.log("training/projection_error", error)

        # continuously log the baseline for a nice transductive plot
        train = Evaluation.kgc_train
        if train in self.evaluations:
            self._log_kgc_results(train, self._kgc_train_results)

        # run evaluations with projections
        evaluations = {Evaluation.kgc_transductive, Evaluation.kgc_inductive}
        for which in evaluations & self.evaluations:
            results = self.run_kgc_evaluation(which)
            self._log_kgc_results(which, results)

        self.clear_projections()

    # /lightning callbacks
    # ----------------------------------------
    # interface

    # batch x tokens x text_dims -> batch x text_dims
    def aggregate(self, encoded: Tensor) -> Tensor:
        raise NotImplementedError()

    # batch x textdims -> [unique] keys x text_dims
    def reduce(self, aggregated, samples):
        raise NotImplementedError()

    # [unique] keys x kge dims
    def project(self, reduced: Tensor) -> Tensor:
        raise NotImplementedError()

    def compare(self, projected: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()

    # /interface
    # ----------------------------------------
    # debugging and introspection

    def str_triple(self, triple: tuple[int]):
        h, r, t = triple

        def format(key):
            kind, idx = self.kgc.idx(key)
            assert kind in {"mid", "vid"}
            name = self.irt2.mentions[idx] if kind == "mid" else self.irt2.vertices[idx]
            return f"{name} ({kind}={key}, {idx=})"

        h = format(h)
        t = format(t)
        r = f"{self.irt2.relations[r]}"

        return f"{h} -{r}- {t}"

    # /debugging and introspection


class SingleAffineProjector(Projector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss = torch.nn.MSELoss()

        self.projector = torch.nn.Linear(
            self.encoder.config.hidden_size,
            self.kgc_embedding_dim,
        )

    def aggregate(self, encoded):
        return encoded[:, 0]

    def reduce(self, aggregated, samples):
        return aggregated, samples

    def project(self, reduced: Tensor):
        return self.projector(reduced)

    def compare(self, projected, target):
        return self.loss(projected, target)
        # return torch.dist(projected, target, p=2) / projected.shape[0]


class MultiAffineProjector(Projector):

    pooling: Literal["mean", "max"]

    def __init__(self, *args, pooling: str = "mean", **kwargs):
        super().__init__(*args, pooling=pooling, **kwargs)

        self.pooling = pooling
        log.info(f"aggregations will be >[{self.pooling}-pooled]<")

        self.loss = torch.nn.MSELoss()
        self.projector = torch.nn.Linear(
            self.encoder.config.hidden_size,
            self.kgc_embedding_dim,
        )

    def aggregate(self, encoded):
        return encoded[:, 0]

    def pool(self, T):  # B x D -> D
        if self.pooling == "max":
            return T.max(axis=0).values

        elif self.pooling == "mean":
            return T.mean(axis=0)

        else:
            assert False, f"unknown pooling strategy {self.pooling}"

    def reduce(self, aggregated, samples):
        # inner-batch indexes
        counter = count()

        keys = {sample.key: sample for sample in samples}
        flat = [sample.key for sample in samples]

        # e.g. given two entities and a batch_size of 5:
        # (8, 8, 8, 7, 7) -> [(8, [0, 1, 2]), (7, [3, 4])]
        grouped = [
            (entity, [next(counter) for _ in grouper])
            for entity, grouper in groupby(flat)
        ]

        # batch x kge_dims -> unique entities x kge_dims
        pooled = [self.pool(aggregated[idxs]) for _, idxs in grouped]
        pooled = torch.vstack(pooled)

        unique_keys = tuple(zip(*grouped))[0]
        return pooled, [keys[key] for key in unique_keys]

    def project(self, reduced: Tensor):
        return self.projector(reduced)

    def compare(self, projected, target):
        return self.loss(projected, target)


MODELS = {
    "single context affine projector": SingleAffineProjector,
    "multi context affine projector": MultiAffineProjector,
}
