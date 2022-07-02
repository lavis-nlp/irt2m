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
    "adamw": torch.optim.AdamW,
}


# --
# functional / shared between models


# batch x dims -> dims
def f_pool(T, reduction: Literal["mean", "max"]):
    if reduction == "max":
        return T.max(axis=0).values

    elif reduction == "mean":
        return T.mean(axis=0)


# batch x dims -> unique samples x dims
def f_reduce_multi(
    aggregated,
    samples,
    reduction: Literal["mean", "max"],
):
    # inner-batch indexes
    counter = count()

    keys = {sample.key: sample for sample in samples}
    flat = [sample.key for sample in samples]

    # e.g. given two entities and a batch_size of 5:
    # (8, 8, 8, 7, 7) -> [(8, [0, 1, 2]), (7, [3, 4])]
    grouped = [
        (entity, [next(counter) for _ in grouper]) for entity, grouper in groupby(flat)
    ]

    # batch x kge_dims -> unique entities x kge_dims
    pooled = [f_pool(aggregated[idxs], reduction) for _, idxs in grouped]
    pooled = torch.vstack(pooled)

    unique_keys = tuple(zip(*grouped))[0]
    return pooled, [keys[key] for key in unique_keys]


# --


class Evaluation(enum.Enum):

    # closed world embeddings
    kgc_train = "kgc/train"

    # closed world projections
    kgc_transductive = "kgc/transductive"

    # open world (validation split) projections
    kgc_inductive = "kgc/inductive"

    # open world (test split) projections
    kgc_test = "kgc/test"


# not using abc to avoid multi-inheritance
class Projector(pl.LightningModule):
    """Base model with common functionality."""

    irt2: IRT2
    kgc: data.KGC
    evaluations: set[Evaluation]

    config: data.Config
    encoder: tf.BertModel
    tokenizer: tf.BertTokenizer

    # We use manual optimization (gradient accumulation for multi-context models)
    # https://pytorch-lightning.readthedocs.io/en/1.5.10/common/optimizers.html
    automatic_optimization = False

    @property
    def debug(self) -> bool:
        return self.config["mode"] in {"probe", "limited"}

    @property
    def subbatch_size(self) -> int:
        return self.trainer.train_dataloader.loaders.subbatch_size

    # --

    def _init_projections(self, total, dim, dtype=None):
        # registering buffers to assure that they are (1) not part
        # of any gradient computation and (2) always on the correct device

        self.register_buffer(
            "projections",
            torch.zeros((total, dim), dtype=dtype),
        )
        self.register_buffer(
            "projections_counts",
            torch.zeros((total,), dtype=torch.int),
        )

        log.info(
            "registered projections buffer: "
            f"{self.projections.shape[0]} x {self.projections.shape[1]} "
            f"({self.projections.dtype})"
        )
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

    def _projection_error(self, targets):
        assert self._gathered_projections

        idxs = self.kgc.closed_world_idxs
        mask = self.projections_counts[idxs] != 0

        projections = self.projections[idxs][mask]
        targets = targets[idxs][mask]

        error = torch.dist(projections, targets) / targets.shape[1]

        return error

    def __init__(
        self,
        irt2: IRT2,
        config: data.Config,
        tokenizer: tf.BertTokenizer,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.irt2 = irt2
        self.config = config
        self.tokenizer = tokenizer

        self.kgc = data.KGC(irt2)
        self.evaluations = {Evaluation(name) for name in config["evaluations"]}
        log.info(f"will run evaluations: {[p.value for p in self.evaluations]}")

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

    # interface

    def forward(self, batch):
        raise NotImplementedError()

    def evaluate_kgc(self, which, **kwargs):
        raise NotImplementedError()

    # kgc

    def run_kgc_evaluation(self, which: Evaluation) -> dict:
        # print(f"\n\n\nevaluate {which.value} {datetime.now()}\n")
        log.info(f"running >[{which.value}]< evaluation")

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

        evaluator = pykeen.evaluation.RankBasedEvaluator(filtered=True)
        results = self.evaluate_kgc(evaluator, which, **kwargs)

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

    # kgc embedding shenanigans

    def _overwrite_embedding(
        self,
        which: Evaluation,
        new: pykeen.nn.Embedding,
        closed_world: torch.Tensor,
    ):

        cw_idxs = self.kgc.closed_world_idxs
        ow_idxs = self.kgc.open_world_idxs

        # train uses the original embeddings
        # (which also checks whether the targets tensor is correct)
        if which is Evaluation.kgc_train:
            idxs, source = cw_idxs, closed_world

        if which is Evaluation.kgc_transductive:
            idxs, source = cw_idxs, self.projections

        if which is Evaluation.kgc_inductive:
            idxs, source = ow_idxs, self.projections

        if which is Evaluation.kgc_test:
            idxs, source = ow_idxs, self.projections

        # pykeen directly modifies the _embedding.weight.data
        # https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/nn/representation.py#L395
        log.info(f"replacing {len(idxs)} embeddings for {which.value}")

        # https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/nn/representation.py#L407
        if torch.is_complex(source):
            source = torch.view_as_real(source).view((source.shape[0], -1))

        new._embeddings.weight[idxs] = source[idxs]

    @contextmanager
    def replace_embeddings(
        self,
        which: Evaluation,
        model: pykeen.models.ERModel,
        old: pykeen.nn.Embedding,
        # closed_world=self.targets in case of KGCProjector
        # closed_world=self.entities in case of JointProjector
        closed_world: torch.Tensor,
    ):

        # https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/models/unimodal/complex.py#L102
        new = old.__class__(
            max_id=self.kgc.num_embeddings,
            shape=old.shape,
            # https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/models/unimodal/complex.py#L106
            dtype=torch.cfloat if old.is_complex else torch.get_default_dtype(),
            regularizer=old.regularizer,
            trainable=False,
        )

        new = new.to(device=self.device)

        # always provide the closed-world embeddings
        self._overwrite_embedding(Evaluation.kgc_train, new, closed_world)

        if which is not Evaluation.kgc_train:
            self._overwrite_embedding(which, new, closed_world)

        model.entity_representations[0] = new

        try:
            yield

        except Exception as exc:
            log.error(f"kgc evaluation error: {exc}")

        finally:
            log.info("restoring original kgc model")
            model.entity_representations[0] = old

    # /kgc embedding shenanigans
    # ----------------------------------------

    def on_fit_end(self):
        log.info("finished fitting")

    # def on_train_start(self):
    #     log.info("starting training")

    def on_train_epoch_start(self):
        log.info(f"starting training >[epoch {self.current_epoch}]<")
        self.log("trainer/global_epoch", self.current_epoch)

    def on_train_epoch_end(self):
        log.info(f"training epoch {self.current_epoch} ended")

    def on_validation_epoch_start(self, *_):
        log.info("validation epoch start")

    def on_fit_start(self):
        print("\n\n")
        print(self.kgc.description)
        print()

        print("\nFITTING\n")
        log.info("starting to fit")

    def on_validation_epoch_end(self, *_):
        log.info("validation epoch end")

        self.gather_projections()

        if self.global_step:

            # log the average distance between projections
            # and targets (somewhat of a validation loss)
            error = self.projection_error()
            self.log("training/projection_error", error)

            # run evaluations with projections
            evaluations = {Evaluation.kgc_transductive, Evaluation.kgc_inductive}
            for which in evaluations & self.evaluations:
                results = self.run_kgc_evaluation(which)
                self._log_kgc_results(which, results)

        self.clear_projections()


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


class KGCProjector(Projector):

    kgc_model: KGCModel

    # see _init_projections
    targets: Tensor
    projections: Tensor
    projections_counts: Tensor

    # ----------------------------------------
    # projection management

    def _init_projections(self):
        # when using a buffer to save reference embeddings, pytorch crashes
        # with "Trying to backward through the graph a second time (or directly
        # access saved tensors after they have already been freed)"
        # self.register_buffer("targets", embedding)

        # init reference embedding
        self.targets = self.cw_entity_embedding._embeddings.weight.detach()
        self.targets = self.targets.to(device=self.device)

        # we require real embeddings for norm/euclidean operations
        assert self.targets.dtype == torch.float32

        _, dim = self.targets.shape
        total = self.kgc.num_embeddings
        # total = self.kgc.dataset.num_entities

        log.info(f"registered target tensor: {self.targets.shape}")
        super()._init_projections(total, dim)

    def evaluate_kgc(self, evaluator, which, **kwargs):
        model = self.kgc_model.model.to(device=self.device)
        old = self.cw_entity_embedding
        closed_world = self.targets

        with self.replace_embeddings(which, model, old, closed_world):
            results = evaluator.evaluate(
                model=model,
                tqdm_kwargs=dict(
                    leave=False,
                    ncols=data.TERM_WIDTH,
                    file=sys.stdout,
                ),
                **kwargs,
            )

        return results

    def projection_error(self):
        return self._projection_error(self.targets)

    # /projection management
    # ----------------------------------------
    # kgc model

    @property
    def kgc_embedding_dim(self) -> int:
        # TODO use self.cw_entity_embedding.shape[0]
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
        super().__init__(irt2, config, *args, **kwargs)
        self.kgc_model = KGCModel(config)
        self._init_projections()

    # ----------------------------------------
    # lightning callbacks and training

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
        log.error(f"move targets tensor to {self.device}")
        self.targets = self.targets.to(device=self.device)
        super().on_fit_start()

        train = Evaluation.kgc_train
        if train in self.evaluations:
            self._kgc_train_results = self.run_kgc_evaluation(train)

    def on_validation_epoch_end(self, *args):
        if self.global_step:

            # continuously log the baseline for a nice transductive plot
            train = Evaluation.kgc_train
            if train in self.evaluations:
                self._log_kgc_results(train, self._kgc_train_results)

        super().on_validation_epoch_end(*args)

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


class SingleAffineProjector(KGCProjector):
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


class MultiAffineProjector(KGCProjector):

    pooling: Literal["mean", "max"]

    def __init__(self, *args, pooling: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)

        self.pooling = pooling
        log.info(f"aggregations will be >[{self.pooling}-pooled]<")

        self.loss = torch.nn.MSELoss()
        self.projector = torch.nn.Linear(
            self.encoder.config.hidden_size,
            self.kgc_embedding_dim,
        )

    def aggregate(self, encoded):
        return encoded[:, 0]

    def reduce(self, aggregated, samples):
        return f_reduce_multi(aggregated, samples, self.pooling)

    def project(self, reduced: Tensor):
        return self.projector(reduced)

    def compare(self, projected, target):
        return self.loss(projected, target)


#
#   JOINT MODELS
#


class JointProjector(Projector):
    kgc: data.KGC

    def __init__(
        self,
        irt2: IRT2,
        config: dict,
        *args,
        embedding_dim: int = None,
        regularizer: str,
        regularizer_kwargs: dict,
        **kwargs,
    ):
        super().__init__(irt2, config, *args, **kwargs)

        self.project = torch.nn.Linear(
            in_features=self.encoder.config.hidden_size,
            # https://github.com/pykeen/pykeen/blob/v1.8.1/src/pykeen/nn/representation.py#L353
            out_features=embedding_dim * 2,
        )

        self.loss = torch.nn.CrossEntropyLoss()

        self.scorer = pykeen.models.unimodal.ComplEx(
            embedding_dim=embedding_dim,
            regularizer=regularizer,
            regularizer_kwargs=regularizer_kwargs,
            triples_factory=self.kgc.closed_world_dataset,
            random_seed=config["seed"],
        )

        self.entities = self.scorer.entity_representations[0]
        self.relations = self.scorer.relation_representations[0]

        # self.score = pykeen.nn.functional.complex_interaction
        log.info(f"initialized triple scorer: {self.scorer.interaction}")

        log.info(
            f"initialized >[entity]< embedding: "
            f"{self.entities.max_id} x {self.entities.shape[0]}"
        )

        log.info(
            f"initialized >[relation]< embedding: "
            f"{self.relations.max_id} x {self.relations.shape[0]}"
        )

        self._init_projections(
            total=self.kgc.num_embeddings,
            dim=embedding_dim,
            dtype=torch.cfloat,
        )

    def _forward_project(self, collation):
        idxs, samples = collation
        B = idxs.shape[0]

        # view_as_complex: N x D x 2 -> N x D
        # view_as_real:    N x D -> N x D x 2
        complex = torch.view_as_complex

        # batch x text_dims
        encs = self.encode(idxs)[:, 0]

        # (unique) samples x text_dims
        self.reduce(encs, samples)

        # (unique) samples x embedding_dims
        encs = complex(self.project(encs).view(B, -1, 2))

        return encs, samples

    def _forward_directed(
        self,
        idxs: torch.Tensor,  # B x tokens
        er: torch.Tensor,  # B x 2
        kind: Literal["hr", "tr"],
        samples: list,
    ):
        if len(idxs) == 0:
            assert samples == []
            return [self.void_c] * 3, self.void_c, tuple()

        # hr -> target head idx, rel idx
        # tr -> target tail idx, rel idx
        assert kind in {"hr", "tr"}

        B = idxs.shape[0]
        E = self.entities.max_id

        # batch x embedding_dims
        encs, samples = self._forward_project((idxs, samples))

        # batch x embedding_dims
        ents = self.entities(None)  # gets all
        rels = self.relations(er[:, 1])

        # batch x num_entities x embedding_dim
        e, r = (
            # for E = 3:
            # [0, 1, 2] -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
            x.repeat_interleave(E, dim=0).view(B, E, -1)
            for x in (encs, rels)
        )

        # batch x num_entities x embedding_dim
        y = torch.stack([ents for _ in range(B)])

        h, r, t = (e, r, y) if kind == "tr" else (y, r, e)
        return (h, r, t), encs, samples

    def _subbatch(self, collation):
        # take the head and tail parts separately
        head, tail = collation[:3], collation[3:]
        assert len(head) == len(tail)

        # figure out the subbatch steps over all samples
        N = len(head[-1]) + len(tail[-1])
        steps = range(0, N, self.subbatch_size)
        offset = len(head[-1])

        # given 3 head samples and 5 tail samples
        # with a subbatch size of 5:
        #   j=0, k=5, l=0, m=2  -> 3 head, 2 tail
        #   j=5, k=8, l=2, m=5  -> 0 head, 3 tail
        for j, k in zip_longest(steps, steps[1:], fillvalue=N):
            l, m = max(0, j - offset), max(0, k - offset)
            s1, s2 = slice(j, k), slice(l, m)

            t1 = tuple(map(lambda h: h[s1], head))
            t2 = tuple(map(lambda t: t[s2], tail))

            yield t1 + t2

    def training_step(
        self,
        collation: tuple[Tensor, list[data.TripleSample]],
        batch_idx,
    ):
        losses = []
        optimizer = self.optimizers()

        for subcollation in self._subbatch(collation):
            # th, tt: batch x tokens (int)
            # hr, tr: batch x 2 (int)
            th, hr, h_samples, tt, tr, t_samples = subcollation

            x_h, h_encs, h_samples = self._forward_directed(th, hr, "hr", h_samples)
            x_t, t_encs, t_samples = self._forward_directed(tt, tr, "tr", t_samples)

            # h, r, t: batch x num_entities x embedding_dim
            # samples: batch
            # encs: batch x embedding_dim
            h, r, t = list(map(torch.cat, zip(x_h, x_t)))
            samples = h_samples + t_samples
            encs = torch.cat((h_encs, t_encs))

            # score them triples
            # batch x num_entities
            scores = self.scorer.interaction(h, r, t)
            targets = torch.cat(
                (
                    hr[:, 0] if len(hr) else self.void_i,
                    tr[:, 0] if len(tr) else self.void_i,
                )
            )

            loss = self.loss(scores, targets)
            losses.append(loss)

            self.manual_backward(loss)
            self.update_projections(encs, samples)

        optimizer.step()
        optimizer.zero_grad()

        loss = torch.stack(losses).mean()
        self.log("training/loss", loss)

        return loss

    # --

    def validation_step(
        self,
        collation: tuple[Tensor, list[data.ProjectorSample]],
        batch_idx,
    ):
        projections, samples = self._forward_project(collation)
        self.update_projections(projections, samples)

    # --

    def projection_error(self):
        return self._projection_error(self.entities(None))

    def evaluate_kgc(self, evaluator, which, **kwargs):
        model = self.scorer
        old = self.entities
        closed_world = self.entities._embeddings.weight

        with self.replace_embeddings(which, model, old, closed_world):
            results = evaluator.evaluate(
                model=model,
                tqdm_kwargs=dict(
                    leave=False,
                    ncols=data.TERM_WIDTH,
                    file=sys.stdout,
                ),
                **kwargs,
            )

        return results

    def on_fit_start(self):
        super().on_fit_start()

        # empty tensors to avoid control structures around
        # head/tail partitions (e.g. see self.training_step)
        self.void_i = torch.zeros(0).to(device=self.device, dtype=torch.int)
        self.void_f = torch.zeros(0).to(device=self.device, dtype=torch.float)
        self.void_c = torch.zeros(0).to(device=self.device, dtype=torch.cfloat)

        # PyKEEN related initialization

    def on_validation_epoch_end(self, *args):
        if self.global_step:
            # continuously log the trained embeddings performance
            train = Evaluation.kgc_train
            if train in self.evaluations:
                results = self.run_kgc_evaluation(train)
                self._log_kgc_results(train, results)

        super().on_validation_epoch_end(*args)


class SingleComplexJoint(JointProjector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reduce(self, aggregated, samples):
        return aggregated, samples


class MultiComplexJoint(JointProjector):
    pooling: Literal["mean", "max"]

    def __init__(self, *args, pooling: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)

        self.pooling = pooling
        log.info(f"aggregations will be >[{self.pooling}-pooled]<")

    def reduce(self, aggregated, samples):
        return f_reduce_multi(aggregated, samples, self.pooling)


MODELS = {
    "single context affine projector": SingleAffineProjector,
    "multi context affine projector": MultiAffineProjector,
    "single context complex joint": SingleComplexJoint,
    "multi context complex joint": MultiComplexJoint,
}
