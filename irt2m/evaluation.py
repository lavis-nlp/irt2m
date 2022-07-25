# -*- coding: utf-8 -*-
"""Evaluate models."""

import abc
import csv
import logging
import pickle
import re
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path
from typing import Optional

import torch
import yaml
from irt2.dataset import IRT2
from irt2.evaluation import RankEvaluator, Ranks
from ktz.collections import dflat
from ktz.collections import drslv as _drslv
from ktz.filesystem import path as kpath
from tabulate import tabulate
from tqdm import tqdm as _tqdm

import irt2m
from irt2m import data, models

log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=data.TERM_WIDTH)
drslv = partial(_drslv, sep=".")


# --


SUBDIR = "evaluation"


def hits(evaluation, dic, k):
    trail = None

    kgc = {
        models.Evaluation.kgc_train.key,
        models.Evaluation.kgc_transductive.key,
        models.Evaluation.kgc_inductive.key,
    }

    if evaluation in kgc:
        trail = f"both.realistic.hits_at_{k}"
    if evaluation == models.Evaluation.irt2_inductive.key:
        trail = f"all.micro.hits_at_{k}"

    assert trail, f"add trail for {evaluation}"
    return drslv(dic, trail, default=-1)


def _apply_backports(config: dict):
    # set defaults for missing options in older models
    if "mode" not in config:
        config["mode"] = "full"


class ProjectorData:

    path: Path
    log: str
    config: data.Config
    monitored_metric: models.Evaluation
    validation: dict[dict[str, dict]]  # kind -> epoch -> results
    checkpoints: list[Path]

    @cached_property
    def description(self):
        header = str(self)
        checkpoints = "\n".join([f"  - {path.stem}" for path in self.checkpoints])

        tablehead = ("epoch", "step") + tuple(self.validation)
        tablerows = []

        for key in self.validation[tablehead[2]]:
            metrics = (
                hits(kind, self.validation[kind][key], k=10) for kind in tablehead[2:]
            )

            tablerows.append(key + tuple(metrics))

        # dsc by h@10, epoch

        sortkey = tablehead.index(self.monitored_metric.key)
        tablerows.sort(key=lambda t: (t[sortkey], t[0]), reverse=True)
        tablerows = tablerows[:5]

        return "\n".join(
            [
                header,
                "\nCheckpoints:",
                checkpoints,
                f"\nValidation using {self.monitored_metric} (Top 5):",
                tabulate(
                    tablerows,
                    headers=map(lambda s: s[:10], tablehead),
                    floatfmt="2.3f",
                ),
            ]
        )

    def best(self, k: int = 10):
        key = self.monitored_metric.key
        vals = self.validation[key]
        gen = ((hits(key, dic, k=k), epoch) for (epoch, _), dic in vals.items())

        top = sorted(gen, reverse=True)
        total = max(epoch for _, epoch in top)

        hatk, epoch = top[0]
        return hatk, epoch, total

    def __str__(self):
        timestamp = datetime.fromisoformat(self.config["timestamp"])
        prefix = self.config["prefix"]
        h10, epoch, total = self.best()

        return (
            f"Projectors {prefix} {timestamp}"
            f" (best h@10: {h10:2.3f} at epoch {epoch}/{total})"
        )

    # loads pre-irt2 evaluation metrics (if the result directory
    # contains a "validation" folder it's ng and if there's a
    # "kg" folder it is legacy
    def _init_legacy_validation(self):
        validation = {"train": {}, "transductive": {}, "inductive": {}}

        log.info("falling back to legacy validation metrics")
        for glob in self.path.glob("kgc/*yaml"):
            # expecting file names like epoch=0-step=0_inductive.yaml
            _, epoch, _, step, kind = re.split(r"[=\-_]", glob.stem)
            with glob.open(mode="r") as fd:
                key = int(epoch), int(step)
                validation[kind][key] = yaml.safe_load(fd)

        self.monitored_metric = models.Evaluation.kgc_inductive
        self.validation = {f"kgc_{k}": v for k, v in validation.items()}

    def _init_validation(self):
        path = self.path / "validation"

        if not path.is_dir():
            self._init_legacy_validation()
            return

        metric = self.config["checkpoint"]["monitor"]
        self.monitored_metric = models.Evaluation.from_metric(metric)

        validation = defaultdict(dict)
        for glob in path.glob("*.yaml"):
            # expecting file names like epoch=0-step=0.yaml
            _, epoch, _, step = re.split(r"[=\-_]", glob.stem)
            key = int(epoch), int(step)

            with glob.open(mode="r") as fd:
                results = yaml.safe_load(fd)
                for kind, metrics in results.items():
                    validation[kind][key] = metrics

        self.validation = dict(validation)

    def __init__(self, path: Path):
        log.info(f"initializing trained projectors from {path}")
        self.path = path

        with (path / "log.txt").open(mode="r") as fd:
            self.log = fd.read()

        with (path / "config.yaml").open(mode="r") as fd:
            self.config = yaml.safe_load(fd)
            _apply_backports(self.config)

        self._init_validation()

        # there's always a "last.ckpt" which we can fall back to
        self.checkpoints = []
        for glob in path.glob("checkpoints/*ckpt"):
            self.checkpoints.append(glob)

        log.info(f"loaded {self}")

    # ---

    def _get_checkpoint(self, checkpoint):
        checkpoints = self.path / "checkpoints"

        if checkpoint is None:

            # legacy: can be removed later
            key = "irt2_inductive"
            key = key if key in self.validation else "kgc_inductive"

            _, epoch, _ = self.best()

            globs = list(checkpoints.glob(f"epoch={epoch}*.ckpt"))
            path = globs[0] if len(globs) == 1 else checkpoints / "last.ckpt"

        else:
            path = checkpoints / checkpoint

        # --

        return path

    def load(self, irt2: IRT2, checkpoint: str = None):
        print("loading projector datamodule...")
        module = data.ProjectorModule(irt2, self.config)

        checkpoint = self._get_checkpoint(checkpoint)
        print(f"loading checkpoint: {checkpoint.name}")

        Projector = models.MODELS[self.config["model"]]
        model = Projector.load_from_checkpoint(
            checkpoint,
            irt2=irt2,
            config=self.config,
            tokenizer=module.tokenizer,
            **self.config.get("model_kwargs", {}),
        )

        return checkpoint, module, model


def create_projections(
    config,
    irt2,
    model,
    module,
    device,
    batch_size,
):
    ds_kwargs = config["module"]["valid_ds_kwargs"]
    print(">> dataset: ", ds_kwargs)

    # evaluate projections like they were trained
    dl_kwargs = config["module"]["train_loader_kwargs"]

    dl_kwargs |= dict(
        shuffle=False,
        batch_size=batch_size,
        subbatch_size=batch_size,
    )

    print(">> dataloader:", dl_kwargs)
    print(">> ! loading all samples from test")

    loaders = {
        "closed-world (train)": data.ProjectorLoader(
            data.VertexFlatDataset(
                irt2=irt2,
                tokenizer=module.tokenizer,
                config=config,
                kind="train",
                **ds_kwargs,
            ),
            collate_fn=data.VertexFlatDataset.collate_fn,
            **dl_kwargs,
        ),
        "open-world (validation)": data.ProjectorLoader(
            data.MentionFlatDataset(
                irt2=irt2,
                tokenizer=module.tokenizer,
                config=config,
                kind="valid",
                **ds_kwargs,
            ),
            collate_fn=data.VertexFlatDataset.collate_fn,
            **dl_kwargs,
        ),
        "open-world (test)": data.ProjectorLoader(
            data.MentionFlatDataset(
                irt2=irt2,
                tokenizer=module.tokenizer,
                config=config,
                kind="test",
                **ds_kwargs | dict(max_contexts_per_sample=None),
            ),
            collate_fn=data.VertexFlatDataset.collate_fn,
            **dl_kwargs,
        ),
    }

    with torch.no_grad():
        model.clear_projections()

        for name, loader in loaders.items():
            print(f"\n{loader.dataset.description}\n\n")

            gen = tqdm(
                loader,
                desc=f"{name}",
                unit=" batches",
                total=len(loader),
            )

            for batch in gen:
                batch = module.transfer_batch_to_device(batch, device, 0)
                result = model.forward(batch)
                model.update_projections(*result)

        model.gather_projections(force=True)
    return model.projections, model.projections_counts


class EvaluationResult(abc.ABC):

    irt2: IRT2
    data: ProjectorData
    checkpoint: Path

    model: models.Projector
    module: data.ProjectorModule

    def __init__(
        self,
        irt2,
        source,
        batch_size: Optional[int] = None,
        irt2_batch_size: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ):
        source = kpath(source, is_dir=True)
        self.data = ProjectorData(path=source)

        prefix = "  "
        print("Loaded:")
        print(textwrap.indent(str(self.data.description), prefix))
        print()

        self.data.config["mode"] = "probe"

        # initialize model and data

        self.irt2 = IRT2.from_dir(irt2)
        _irt2 = IRT2.from_dir(self.data.config["irt2"])
        assert _irt2.name == self.irt2.name, "wrong IRT2 dataset"

        self.checkpoint, self.module, self.model = self.data.load(self.irt2, checkpoint)
        device = device = torch.device("cuda")

        self.model = self.model.to(device)
        self.model.ensure_device()
        self.model.eval()

        if batch_size is None:
            batch_size = drslv(
                self.data.config, "module.valid_loader_kwargs.batch_size"
            )

        if irt2_batch_size is not None:
            eval_irt2_conf = drslv(
                self.data.config,
                "evaluations_kwargs.irt2",
                default=None,
            )

            if eval_irt2_conf is not None:
                eval_irt2_conf["batch_size"] = irt2_batch_size

        cached = self.init(device, batch_size)
        self.run_evaluation(force=not cached)

    @abc.abstractmethod
    def init(device, batch_size):
        pass

    @abc.abstractmethod
    def run_evaluation(force: bool = False):
        pass

    # --

    def _irt2_rows(self, results, ks):
        rows = []

        ordered = [
            eval.key
            for eval in [
                models.Evaluation.irt2_inductive,
                models.Evaluation.irt2_test,
            ]
        ]

        for key in ordered:
            metrics = results[key]
            prefix = "all.micro"

            hits = tuple(drslv(metrics, f"{prefix}.hits_at_{k}") * 100 for k in ks)
            mrr = drslv(metrics, f"{prefix}.mrr") * 100

            rows.append((key,) + hits + (mrr,))

        return rows


class LinkingEvaluationResult(EvaluationResult):
    def init(self, device, batch_size):
        name = f"{self.checkpoint.stem}.linking.projections.pkl"
        cache = kpath(self.data.path / SUBDIR, create=True) / name

        if cache.exists():
            print("loading projections from cache")

            with cache.open(mode="rb") as fd:
                P, PC = pickle.load(fd)

            self.model.projections = P
            self.model.projections_counts = PC
            self.model._gathered_projections = True

            cached = True

        else:
            print("cache miss! creating projections from text")

            P, PC = create_projections(
                self.data.config,
                self.irt2,
                self.model,
                self.module,
                device,
                batch_size,
            )

            with cache.open(mode="wb") as fd:
                pickle.dump((P, PC), fd)

            cached = False

        mask = PC != 0
        print(f"obtained {mask.sum().item()} projections: {P.shape}")

        return cached

    def run_evaluation(self, force: bool = False):
        # both run pykeen and irt2 evaluations
        path = kpath(self.data.path / SUBDIR, create=True)
        cache = path / f"{self.checkpoint.stem}.metrics.linking.yaml"

        if not force and cache.exists():
            print(f"loading linking results from {cache.name}")

            with cache.open(mode="r") as fd:
                results = yaml.safe_load(fd)
                self.linking_results = results
                return

        print("cache miss! running linking evaluation")

        evaluations = [
            models.Evaluation.irt2_inductive,
            models.Evaluation.irt2_test,
            models.Evaluation.kgc_train,
            models.Evaluation.kgc_transductive,
            models.Evaluation.kgc_inductive,
            models.Evaluation.kgc_test,
        ]

        results = {}
        for which in evaluations:
            kwargs = {}
            if which.evaluator == "irt2":
                name = f"{self.checkpoint.stem}.scores.{which.split}.h5"
                kwargs = dict(out=path / name)

            res = self.model.run_kgc_evaluation(which, **kwargs)
            results[which.key] = res.to_dict()

        with cache.open(mode="w") as fd:
            print(f"saving ranking results to {cache.name}")
            yaml.safe_dump(results, fd)

        self.linking_results = results

    def __str__(self):
        trail = "irt2_test.all.micro.hits_at_10"
        linking_hits = drslv(self.linking_results, trail)

        return (
            f"Results of {self.data.config['prefix']} {self.checkpoint.stem}:"
            f"  Linking {linking_hits * 100:2.3f}"
        )

    @property
    def table(self) -> str:
        rows = []

        # irt2

        rows += self._irt2_rows(self.linking_results, ks=(1, 10))

        # pykeen

        ordered = [
            eval.key
            for eval in [
                models.Evaluation.kgc_train,
                models.Evaluation.kgc_transductive,
                models.Evaluation.kgc_inductive,
                models.Evaluation.kgc_test,
            ]
        ]

        for key in ordered:
            metrics = self.linking_results[key]
            prefix = "both.realistic"
            rows.append(
                (
                    key,
                    drslv(metrics, f"{prefix}.hits_at_1") * 100,
                    drslv(metrics, f"{prefix}.hits_at_10") * 100,
                    drslv(metrics, f"{prefix}.inverse_harmonic_mean_rank") * 100,
                )
            )

        table = tabulate(
            rows,
            headers=["", "both h@1", "both h@10", "both mrr"],
            floatfmt="2.3f",
        )

        return textwrap.indent("\n" + table, prefix="  ")


def _add_predictions(preds, score_batch, relations, targets, vid2idx, pred_filter):
    # add both head and tail predictions
    for score_dic in score_batch:
        for direction, (sample, scoremat) in score_dic.items():

            # (!) this is confusing but to solve the head task we
            # need to use the tail predictions of the kgc models
            direction = "head" if direction == "tail" else "tail"

            probs = torch.nn.functional.softmax(scoremat, dim=1)
            gt = pred_filter[direction]

            for task in gt:
                dic = preds[direction][task]

                vid, rel = task
                score = probs[rel][vid2idx[vid]].item()

                # skip already predicted nodes if the score is lower
                if sample.key in dic and dic[sample.key] < score:
                    continue

                dic[sample.key] = score


def create_predictions(
    config,
    irt2,
    model,
    module,
    device,
    batch_size,
):
    ds_kwargs = config["module"]["valid_ds_kwargs"]
    print(">> dataset: ", ds_kwargs)

    # evaluate projections like they were trained
    dl_kwargs = config["module"]["train_loader_kwargs"]

    dl_kwargs |= dict(
        shuffle=False,
        batch_size=batch_size,
        subbatch_size=batch_size,
    )

    print(">> dataloader:", dl_kwargs)
    print(">> loading all samples from test!")

    loaders = {
        models.Evaluation.irt2_inductive.key: data.ProjectorLoader(
            data.MentionFlatDataset(
                irt2=irt2,
                tokenizer=module.tokenizer,
                config=config,
                kind="valid",
                **ds_kwargs,
            ),
            collate_fn=data.VertexFlatDataset.collate_fn,
            **dl_kwargs,
        ),
        models.Evaluation.irt2_test.key: data.ProjectorLoader(
            data.MentionFlatDataset(
                irt2=irt2,
                tokenizer=module.tokenizer,
                config=config,
                kind="test",
                **ds_kwargs | dict(max_contexts_per_sample=None),
            ),
            collate_fn=data.VertexFlatDataset.collate_fn,
            **dl_kwargs,
        ),
    }

    prediction = {
        name: dict(
            head=defaultdict(dict),
            tail=defaultdict(dict),
        )
        for name in loaders
    }

    targets = model.kgc.closed_world_idxs.to(device=device)
    vid2idx = {vid: idx for idx, vid in enumerate(targets.tolist())}

    # we only need to retain predictions
    # for the tasks we want to evaluate
    pred_filter = {
        models.Evaluation.irt2_inductive.key: dict(
            head=set(irt2.open_ranking_val_heads),
            tail=set(irt2.open_ranking_val_tails),
        ),
        models.Evaluation.irt2_test.key: dict(
            head=set(irt2.open_ranking_test_heads),
            tail=set(irt2.open_ranking_test_tails),
        ),
    }

    with torch.no_grad():
        model.clear_projections()
        relations = model.kgc.relation_idxs

        for name, loader in loaders.items():
            print(f"\n{loader.dataset.description}\n\n")

            gen = tqdm(
                loader,
                desc=f"{name}",
                unit=" batches",
                total=len(loader),
            )

            for batch in gen:
                batch = module.transfer_batch_to_device(batch, device, 0)
                score_batch = model.score(batch, relations, targets)

                _add_predictions(
                    prediction[name],
                    score_batch,
                    relations,
                    targets,
                    vid2idx,
                    pred_filter[name],
                )

    return prediction


class RankingEvaluationResult(EvaluationResult):

    # TODO assert not masked

    def init(self, device, batch_size):
        assert not drslv(self.data.config, "module.train_ds_kwargs.masking")

        name = f"{self.checkpoint.stem}.ranking.predictions.pkl"
        cache = kpath(self.data.path / SUBDIR, create=True) / name

        if cache.exists():
            print("loading ranking predictions from cache")

            with cache.open(mode="rb") as fd:
                self.preds = pickle.load(fd)
                cached = True

        else:
            print("cache miss! create predictions")

            self.preds = create_predictions(
                self.data.config,
                self.irt2,
                self.model,
                self.module,
                device,
                batch_size,
            )

            with cache.open(mode="wb") as fd:
                pickle.dump(self.preds, fd)

            cached = False

        return cached

    def run_evaluation(self, force: bool = False):
        path = kpath(self.data.path / SUBDIR, create=True)
        cache = path / f"{self.checkpoint.stem}.metrics.ranking.yaml"

        if not force and cache.exists():
            print(f"loading ranking results from {cache.name}")

            with cache.open(mode="r") as fd:
                results = yaml.safe_load(fd)
                self.ranking_results = results
                return

        print("cache miss! running ranking evaluation")

        # ---

        tqdm_kwargs = dict(
            leave=False,
            ncols=data.TERM_WIDTH,
            file=sys.stdout,
        )

        valid_key = models.Evaluation.irt2_inductive.key
        test_key = models.Evaluation.irt2_test.key

        mapping = (
            (valid_key, "head", self.irt2.open_ranking_val_heads),
            (valid_key, "tail", self.irt2.open_ranking_val_tails),
            (test_key, "head", self.irt2.open_ranking_test_heads),
            (test_key, "tail", self.irt2.open_ranking_test_tails),
        )

        rankdic = defaultdict(dict)
        for split, direction, gt in mapping:
            print(f"adding ranks for {split} - {direction=}...")

            preds = self.preds[split][direction]
            assert set(preds) == set(gt)

            ranks = Ranks(gt).add_iter(
                # important: use generator to keep RAM usage low
                ((task, scores.items()) for task, scores in preds.items()),
                progress=True,
                progress_kwargs=tqdm_kwargs,
            )

            rankdic[split][direction] = (ranks, gt)

        results = {}
        for split, ranks in rankdic.items():
            evaluator = RankEvaluator(**ranks)
            results[split] = evaluator.compute_metrics(ks=(10, 100))

        with cache.open(mode="w") as fd:
            print(f"saving ranking results to {cache.name}")
            yaml.safe_dump(results, fd)

        self.ranking_results = results

    @property
    def table(self) -> str:
        ks = (100,)

        rows = self._irt2_rows(self.ranking_results, ks=ks)
        table = tabulate(
            rows,
            headers=[""] + [f"both h@{k}" for k in ks] + ["both mrr"],
            floatfmt="2.3f",
        )

        return textwrap.indent("\n" + table, prefix="  ")

    def __str__(self):
        trail = "irt2_test.all.micro.hits_at_100"
        ranking_hits = drslv(self.ranking_results, trail)

        return (
            f"Ranking of {self.data.config['prefix']} {self.checkpoint.stem}:"
            f" RANKING {ranking_hits * 100:2.3f} hits@100"
        )

    # ---


def linking(
    irt2: str,
    source: str,
    batch_size: Optional[int] = None,
    irt2_batch_size: Optional[int] = None,
    checkpoint: Optional[str] = None,
):
    print(irt2m.banner)
    log.info(f"run evaluation for {source}")

    result = LinkingEvaluationResult(
        irt2,
        source,
        batch_size=batch_size,
        irt2_batch_size=irt2_batch_size,
        checkpoint=checkpoint,
    )

    print("\nLINKING TASK")
    print(result.table)


def ranking(
    irt2: str,
    source: str,
    batch_size: Optional[int] = None,
    checkpoint: Optional[str] = None,
):
    print(irt2m.banner)
    log.info(f"run evaluation for {source}")

    result = RankingEvaluationResult(
        irt2,
        source,
        batch_size=batch_size,
        checkpoint=checkpoint,
    )

    print("\nRANKING TASK")
    print(result.table)


# ---


def _dic2row(dic, mapping):
    return {k: drslv(dic, v, default=0) for k, v in mapping.items()}


def _irt2row(dic, key, *contains):
    return {
        f"{key} {k}".replace("hits_at_", "h@"): v * 100
        for k, v in dflat(dic[key]).items()
        if all(c in k for c in contains)
    }


def _prefix_row(result):
    return {
        "irt2": result.irt2.name,
        "model": result.model.__class__.__name__,
    }


def _config_row(result):
    additional = {
        "checkpoint": result.checkpoint.stem,
    }

    config = _dic2row(
        result.data.config,
        {
            "prefix": "prefix",
            "date": "timestamp",
            "contexts per sample": "module.train_ds_kwargs.max_contexts_per_sample",
            "contexts per batch": "module.train_ds_kwargs.contexts_per_sample",
            "batch size": "module.train_loader_kwargs.batch_size",
            "subbatch size": "module.train_loader_kwargs.subbatch_size",
            "epochs": "trainer.max_epochs",
        },
    )

    return additional | config


def _report_linking_row(result):
    row = {}

    row |= _prefix_row(result)
    row |= _irt2row(result.linking_results, "irt2_test", "micro")
    row |= _irt2row(result.linking_results, "irt2_inductive", "micro")
    row |= _config_row(result)

    linking_row = _dic2row(
        result.linking_results,
        {
            "kgc_test h@1": "kgc_test.both.realistic.hits_at_1",
            "kgc_test h@10": "kgc_test.both.realistic.hits_at_10",
            "kgc_inductive h@1": "kgc_inductive.both.realistic.hits_at_1",
            "kgc_inductive h@10": "kgc_inductive.both.realistic.hits_at_10",
            "kgc_transductive h@1": "kgc_transductive.both.realistic.hits_at_1",
            "kgc_transductive h@10": "kgc_transductive.both.realistic.hits_at_10",
            "kgc_train h@1": "kgc_train.both.realistic.hits_at_1",
            "kgc_train h@10": "kgc_train.both.realistic.hits_at_10",
        },
    )
    row |= {k: v * 100 for k, v in linking_row.items()}

    row |= _irt2row(result.linking_results, "irt2_test", "macro")
    row |= _irt2row(result.linking_results, "irt2_inductive", "macro")

    # supplemental

    row |= {
        "folder": str(result.data.path),
        "evaluated on": datetime.now().isoformat(),
    }

    return row


def _report_ranking_row(result):
    row = {}

    row |= _prefix_row(result)
    row |= _irt2row(result.ranking_results, "irt2_test", "micro")
    row |= _irt2row(result.ranking_results, "irt2_inductive", "micro")
    row |= _config_row(result)
    row |= _irt2row(result.ranking_results, "irt2_test", "macro")
    row |= _irt2row(result.ranking_results, "irt2_inductive", "macro")

    # supplemental

    row |= {
        "folder": str(result.data.path),
        "evaluated on": datetime.now().isoformat(),
    }

    return row


def create_report(
    batch_size: Optional[int] = None,
    irt2_batch_size: Optional[int] = None,
):
    # folder = kpath(folder, is_dir=True)
    # glob = list(map(lambda p: p.parent, folder.glob("**/checkpoints")))

    # TODO hard-coded for now, change later
    sources = dict(
        ranking=(
            # PSAC
            "data/evaluation/PSAC-T/2022-06-27_18-07-43",
            "data/evaluation/PSAC-S/2022-06-28_02-41-19",
            "data/evaluation/PSAC-M/2022-06-28_11-10-17",
            "data/evaluation/PSAC-L/2022-06-27_18-20-47",
            # PMAC
            "data/evaluation/PMAC-T/2022-06-28_09-54-58",
            "data/evaluation/PMAC-S/2022-06-28_14-08-31",
            "data/evaluation/PMAC-M/2022-06-28_23-47-21",
            "data/evaluation/PMAC-L/2022-06-29_14-30-56",
            # JSC
            "data/evaluation/JSC-T/2022-07-15_16-28-15",
            "data/evaluation/JSC-S/2022-07-18_01-46-16",
            "data/evaluation/JSC-M/2022-07-17_06-15-30",
            "data/evaluation/JSC-L/2022-07-15_12-03-25",
        ),
        linking=(
            # PSAC
            "data/evaluation/PSAC-T/2022-06-27_18-07-43",
            "data/evaluation/PSAC-S/2022-06-28_02-41-19",
            "data/evaluation/PSAC-M/2022-06-28_11-10-17",
            "data/evaluation/PSAC-L/2022-06-27_18-20-47",
            # PMAC
            "data/evaluation/PMAC-T/2022-06-28_09-52-54",
            "data/evaluation/PMAC-S/2022-07-11_17-47-24",
            "data/evaluation/PMAC-M/2022-06-28_14-35-23",
            "data/evaluation/PMAC-L/2022-06-29_14-30-56",
            # JSC
            "data/evaluation/JSC-T/2022-07-15_11-51-35",
            "data/evaluation/JSC-S/2022-07-18_09-37-00",
            "data/evaluation/JSC-M/2022-07-17_06-15-30",
            "data/evaluation/JSC-L/2022-07-15_12-03-25",
        ),
    )

    dataset_map = {
        "L": "large",
        "M": "medium",
        "S": "small",
        "T": "tiny",
    }

    results = dict(linking=[], ranking=[])
    for task in sources:
        for source in sources[task]:
            source = kpath(source, is_dir=True)

            dataset = "irt2-cde-" + dataset_map[source.parent.name[-1]]
            dataset = kpath(
                irt2m.ENV.DIR.DATA / "irt2" / (dataset + f"-{task}"),
                is_dir=True,
            )

            if task == "linking":
                Result = LinkingEvaluationResult
                report = _report_linking_row
            if task == "ranking":
                Result = RankingEvaluationResult
                report = _report_ranking_row

            result = Result(
                irt2=dataset,
                source=source,
                batch_size=batch_size,
            )

            print(textwrap.indent(str(result), prefix="  - "))
            results[task].append(report(result))

    for task, rows in results.items():
        out = irt2m.ENV.DIR.DATA / "evaluation" / f"summary.{task}.csv"
        with (out).open(mode="w") as fd:
            writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"wrote {out}")
