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


def hits(dic, k):
    try:
        return dic["both"]["realistic"][f"hits_at_{k}"]
    except KeyError:
        return 0.0


def _apply_backports(config: dict):
    # set defaults for missing options in older models
    if "mode" not in config:
        config["mode"] = "full"


class ProjectorData:

    path: Path
    log: str
    config: data.Config
    validation: dict[dict[str, dict]]  # kind -> epoch -> results
    checkpoints: list[Path]

    @cached_property
    def description(self):
        header = str(self)
        checkpoints = "\n".join([f"  - {path.stem}" for path in self.checkpoints])
        legacy = "irt2_inductive" not in self.validation

        tablehead = "kgc_train", "kgc_transductive", "kgc_inductive"
        if not legacy:
            tablehead += ("irt2_inductive",)

        tablerows = []
        for key in self.validation[tablehead[0]]:
            row = tuple(hits(self.validation[kind][key], k=10) for kind in tablehead)
            tablerows.append(key + row)

        # kgc_inductive if legacy else irt2_inductive
        sortkey = 4 if legacy else 5

        tablerows.sort(key=lambda t: t[sortkey], reverse=True)  # dsc by h@10
        # tablerows.sort(key=lambda t: (t[0], t[1]))  # asc by epoch, step
        tablerows = tablerows[:5]

        tablehead = ("epoch", "step") + tablehead

        return "\n".join(
            [
                header,
                "\nCheckpoints:",
                checkpoints,
                "\nValidation (Top 5):",
                tabulate(
                    tablerows,
                    headers=map(lambda s: s[:10], tablehead),
                    floatfmt="2.3f",
                ),
            ]
        )

    def best(self, kind: str, k: int = 10):
        vals = self.validation[kind]
        gen = ((hits(dic, k=k), epoch) for (epoch, _), dic in vals.items())

        top = sorted(gen, reverse=True)
        total = max(epoch for _, epoch in top)

        h10, epoch = top[0]
        return h10, epoch, total

    def __str__(self):
        timestamp = datetime.fromisoformat(self.config["timestamp"])
        prefix = self.config["prefix"]

        # kgc_inductive: legacy (can be removed later)
        key = "irt2_inductive"
        key = key if key in self.validation else "kgc_inductive"

        h10, epoch, total = self.best(key)

        return (
            f"Projectors {prefix} {timestamp}"
            f" (best h@10: {h10:2.3f} at epoch {epoch}/{total})"
        )

    def _init_legacy_validation(self):
        validation = {"train": {}, "transductive": {}, "inductive": {}}

        log.info("falling back to legacy validation metrics")
        for glob in self.path.glob("kgc/*yaml"):
            # expecting file names like epoch=0-step=0_inductive.yaml
            _, epoch, _, step, kind = re.split(r"[=\-_]", glob.stem)
            with glob.open(mode="r") as fd:
                key = int(epoch), int(step)
                validation[kind][key] = yaml.safe_load(fd)

        self.validation = {f"kgc_{k}": v for k, v in validation.items()}

    def _init_validation(self):
        path = self.path / "validation"

        if not path.is_dir():
            self._init_legacy_validation()
            return

        self.validation = defaultdict(dict)
        for glob in path.glob("*.yaml"):
            # expecting file names like epoch=0-step=0.yaml
            _, epoch, _, step = re.split(r"[=\-_]", glob.stem)
            key = int(epoch), int(step)

            with glob.open(mode="r") as fd:
                results = yaml.safe_load(fd)
                for kind, metrics in results.items():
                    self.validation[kind][key] = metrics

        self.validation = dict(self.validation)

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

            _, epoch, _ = self.best(key)

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
    print(">> ! LOADING ALL SAMPLES FROM TEST")

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
            gen = tqdm(
                loader,
                desc=f"{name}",
                unit=" batches",
                total=len(loader),
            )

            print(f"\n{loader.dataset.description}\n\n")

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
            batch_size = self.data.config["module"]["valid_loader_kwargs"]["batch_size"]

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
            print("loading linking results from cache")

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

            res = self.model.run_linking_evaluation(which, **kwargs)
            results[which.key] = res.to_dict()

        with cache.open(mode="w") as fd:
            yaml.safe_dump(results, fd)

        self.linking_results = results

    def __str__(self):
        trail = "irt2_test.all.micro.hits_at_10"
        linking_hits = drslv(self.linking_results, trail)

        return (
            f"Ranking of {self.data.config['prefix']} {self.checkpoint.stem}:"
            f" LINKING {linking_hits * 100:2.3f}"
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
            metrics = self.kgc_results[key]
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


def _add_predictions(preds, score_batch, relations, targets):
    # add both head and tail predictions
    for score_dic in score_batch:
        for direction, (sample, scoremat) in score_dic.items():
            probs = torch.nn.functional.softmax(scoremat, dim=1)

            # TODO slow!!
            for i, rel in enumerate(relations.tolist()):
                for vid, score in zip(targets.tolist(), probs[i].tolist()):
                    dic = preds[direction][(vid, rel)]

                    # skip already predicted nodes if the score is lower
                    if sample.key in dic and dic[sample.key] < score:
                        continue

                    dic[sample.key] = score

            # top1 = torch.argmax(probs, dim=1)
            # dim0 = torch.arange(probs.shape[0])

            # ranking head task: 80=hong kong, 1=country of citizenship
            # -> find people living in hong kong (?, 1, 80)

            # ranking tail task: 968=bob kaufman 3=place of birth
            # -> what's the birthplace of bob kaufman

            # vids = targets[top1].tolist()
            # scores = probs[dim0, top1].tolist()
            # rels = relations.tolist()

            # for vid, rel, score in zip(vids, rels, scores):
            #     dic = preds[direction][(vid, rel)]

            #     # skip already predicted nodes if the score is lower
            #     if sample.key in dic and dic[sample.key] < score:
            #         continue

            #     dic[sample.key] = score


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

    prediction = {
        name: dict(
            head=defaultdict(dict),
            tail=defaultdict(dict),
        )
        for name in loaders
    }

    targets = model.kgc.closed_world_idxs.to(device=device)

    with torch.no_grad():
        model.clear_projections()
        relations = model.kgc.relation_idxs

        for name, loader in loaders.items():
            gen = tqdm(
                loader,
                desc=f"{name}",
                unit=" batches",
                total=len(loader),
            )

            print(f"\n{loader.dataset.description}\n\n")
            for batch in gen:
                batch = module.transfer_batch_to_device(batch, device, 0)
                score_batch = model.score(batch, relations, targets)
                _add_predictions(
                    prediction[name],
                    score_batch,
                    relations,
                    targets,
                )

    return {
        name: {  # name: validation, test
            direction: {  # direction: head, tail
                # defaultdict[dict] -> dict[tuple] (for irt2.evaluation.Ranks)
                task: tuple(tups.items())
                for task, tups in preds.items()
            }
            for direction, preds in dic.items()
        }
        for name, dic in prediction.items()
    }


class RankingEvaluationResult(EvaluationResult):

    # TODO assert not masked

    def init(self, device, batch_size):
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

            # (!) this is confusing but to solve the head task we
            # need to use the tail predictions of the kgc models
            direction = "head" if direction == "tail" else "tail"
            preds = self.preds[split][direction]

            ranks = Ranks(gt).add_dict(
                {task: tups for task, tups in preds.items() if task in gt},
                progress=True,
                progress_kwargs=tqdm_kwargs,
            )

            rankdic[split][direction] = (ranks, gt)

        results = {}
        for split, ranks in rankdic.items():
            evaluator = RankEvaluator(**ranks)
            results[split] = evaluator.compute_metrics(ks=(100,))

        self.ranking_results = results

    @property
    def table(self) -> str:
        ks = (100,)

        rows = self._irt2_rows(self.ranking_results, ks=ks)
        table = tabulate(
            rows,
            headers=[""] + ["both h@k" for k in ks] + ["both mrr"],
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
    checkpoint: Optional[str] = None,
):
    print(irt2m.banner)
    log.info(f"run evaluation for {source}")

    result = LinkingEvaluationResult(
        irt2,
        source,
        batch_size=batch_size,
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


def create_report(folder: str):
    assert False, "TODO LINKINGEvaluationResult and RankingEvaluationResult"

    folder = kpath(folder, is_dir=True)
    glob = list(map(lambda p: p.parent, folder.glob("**/checkpoints")))

    print(f"loading {len(glob)} results")

    scenarios = dict(linking=[], ranking=[])
    for source in glob:
        source = kpath(source, is_dir=True)
        result = EvaluationResult(source)

        print(textwrap.indent(str(result), prefix="  - "))

        scenarios["linking"].append(_report_linking_row(result))
        scenarios["ranking"].append(_report_ranking_row(result))

    for scenario, rows in scenarios.items():
        out = folder / f"{scenario}.summary.csv"
        with (out).open(mode="w") as fd:
            writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"wrote {out}")
