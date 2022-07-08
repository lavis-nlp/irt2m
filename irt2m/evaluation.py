# -*- coding: utf-8 -*-
"""Evaluate models."""

import csv
import logging
import pickle
import re
import textwrap
from collections import defaultdict
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path
from typing import Optional

import h5py
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

        tablehead = "train", "transductive", "inductive"
        tablerows = []
        for key in self.validation[tablehead[0]]:
            row = tuple(hits(self.validation[kind][key], k=10) for kind in tablehead)
            tablerows.append(key + row)

        tablerows.sort(key=lambda t: t[4], reverse=True)  # dsc by h@10
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
                    headers=map(lambda s: s[:5], tablehead),
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
        h10, epoch, total = self.best("inductive")

        return (
            f"Projectors {prefix} {timestamp}"
            f" (best h@10: {h10:2.3f} at epoch {epoch}/{total})"
        )

    def __init__(self, path: Path):
        log.info(f"initializing trained projectors from {path}")
        self.path = path

        with (path / "log.txt").open(mode="r") as fd:
            self.log = fd.read()

        with (path / "config.yaml").open(mode="r") as fd:
            self.config = yaml.safe_load(fd)
            _apply_backports(self.config)

        # load all validation results (which were PyKEEN RankingResults)
        self.validation = {"train": {}, "transductive": {}, "inductive": {}}
        for glob in path.glob("kgc/*yaml"):
            # expecting file names like epoch=0-step=0_inductive.yaml
            _, epoch, _, step, kind = re.split(r"[=\-_]", glob.stem)
            with glob.open(mode="r") as fd:
                key = int(epoch), int(step)
                self.validation[kind][key] = yaml.safe_load(fd)

        # there's always a "last.ckpt" which we can fall back to
        self.checkpoints = []
        for glob in path.glob("checkpoints/*ckpt"):
            self.checkpoints.append(glob)

        # path_kgc_results = path / SUBDIR / "kgc.metrics.yaml"
        # self.kgc_results = {}
        # if path_kgc_results.exists():
        #     log.info("found and loading kgc results")
        #     with path_kgc_results.open(mode="r") as fd:
        #         self.kgc_results = yaml.safe_load(fd)

        # path_ranking_results = path / SUBDIR / "ranking.metrics.yaml"
        # self.ranking_results = {}
        # if path_ranking_results.exists():
        #     log.info("found and loading ranking results")
        #     with path_ranking_results.open(mode="r") as fd:
        #         self.ranking_results = yaml.safe_load(fd)

        log.info(f"loaded {self}")

    # ---

    def _get_checkpoint(self, checkpoint):
        checkpoints = self.path / "checkpoints"

        if checkpoint is None:
            _, epoch, _ = self.best("inductive")
            globs = list(checkpoints.glob(f"epoch={epoch}*.ckpt"))
            path = globs[0] if len(globs) == 1 else checkpoints / "last.ckpt"

        else:
            path = checkpoints / checkpoint

        # --

        return path

    def load(self, checkpoint: str = None):
        log.info("loading checkpoint")

        print("loading IRT2...")
        irt2 = IRT2.from_dir(self.config["irt2"])
        print(f"loaded {irt2}")

        print("loading module...")
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

        return irt2, checkpoint, module, model


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
                **ds_kwargs,
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


class EvaluationResult:

    irt2: IRT2
    data: ProjectorData
    checkpoint: Path

    model: models.Projector
    module: data.ProjectorModule

    kgc_results: dict
    ranking_results: dict

    def _init_projections(self, device, batch_size):
        name = f"{self.checkpoint.stem}.projections.pkl"
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

    def _run_kgc_evaluation(self, force: bool = False):
        # both run pykeen and irt2 evaluations
        path = kpath(self.data.path / SUBDIR, create=True)
        cache = path / f"{self.checkpoint.stem}.metrics.kgc.yaml"

        if not force and cache.exists():
            print("loading kgc results from cache")

            with cache.open(mode="r") as fd:
                results = yaml.safe_load(fd)
                self.kgc_results = results
                return

        print("cache miss! running kgc evaluation")

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
            yaml.safe_dump(results, fd)

        self.kgc_results = results

    def _create_ranks(self, gt, task, scores):
        # TODO slow and memory expensive!

        predictions = defaultdict(set)
        for i, (idx, rid) in enumerate(task):
            mid = self.model.kgc.idx2mid[idx]

            for vid, score in enumerate(scores[i]):
                if (vid, rid) in gt:
                    predictions[(vid, rid)].add((mid, score))

        return predictions

    def _run_ranking_evaluation(self, force: bool = False):
        # assumes run_kgc_evaluation was executed before
        # there exists a file with all rankings now
        path = kpath(self.data.path / SUBDIR, create=True)
        cache = path / f"{self.checkpoint.stem}.metrics.ranking.yaml"
        if not force and cache.exists():
            print("loading ranking results from cache")

            with cache.open(mode="r") as fd:
                results = yaml.safe_load(fd)
                self.ranking_results = results
                return

        print("cache miss! running ranking evaluation")

        # (!) attention: the kgc head task corresponds
        # to the ranking tails task
        tasks = {
            models.Evaluation.irt2_inductive: dict(
                head=self.irt2.open_ranking_val_tails,
                tail=self.irt2.open_ranking_val_heads,
            ),
            models.Evaluation.irt2_test: dict(
                head=self.irt2.open_ranking_test_tails,
                tail=self.irt2.open_ranking_test_heads,
            ),
        }

        results = {}
        for which in tasks:
            name = f"{self.checkpoint.stem}.scores.{which.split}.h5"
            path = self.data.path / SUBDIR / name
            h5 = h5py.File(path, "r")

            rank_kwargs = {}
            for kind, grp in ((kind, h5[kind]) for kind in ("head", "tail")):

                logits = torch.from_numpy(grp["scores"][:])  # N x T
                probs = torch.nn.functional.softmax(logits, dim=1)  # N x T
                scores = (logits * probs).numpy()
                print(f"  - {which.split} {kind} for {scores.shape} scores")

                gt = tasks[which][kind]
                predictions = self._create_ranks(gt, grp["tasks"][:], scores)
                ranks = Ranks(gt).add_dict(predictions)
                print(f"  - got {len(predictions)} predictions")

                rank_kwargs[kind] = (ranks, gt)
            results[which.key] = RankEvaluator(**rank_kwargs).compute_metrics()

        with cache.open(mode="w") as fd:
            yaml.safe_dump(results, fd)

        self.ranking_results = results

    def __init__(
        self,
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

        self.irt2, self.checkpoint, self.module, self.model = self.data.load(checkpoint)
        device = device = torch.device("cuda")

        self.model = self.model.to(device)
        self.model.ensure_device()
        self.model.eval()

        if batch_size is None:
            batch_size = self.data.config["module"]["valid_loader_kwargs"]["batch_size"]

        cached = self._init_projections(device, batch_size)
        self._run_kgc_evaluation(force=not cached)
        self._run_ranking_evaluation(force=not cached)

    # ---

    def __str__(self):
        trail = "irt2_test.all.micro.hits_at_10"
        kgc_hits = drslv(self.kgc_results, trail)
        ranking_hits = drslv(self.ranking_results, trail)

        return (
            f"Ranking of {self.data.config['prefix']} {self.checkpoint.stem}:"
            f" KGC {kgc_hits * 100:2.3f}"
            f" Ranking {ranking_hits * 100:2.3f}"
        )

    def _irt2_rows(self, results):
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
            rows.append(
                (
                    key,
                    drslv(metrics, f"{prefix}.hits_at_1") * 100,
                    drslv(metrics, f"{prefix}.hits_at_10") * 100,
                    drslv(metrics, f"{prefix}.mrr") * 100,
                )
            )

        return rows

    @property
    def kgc_table(self) -> str:
        rows = []

        # irt2

        rows += self._irt2_rows(self.kgc_results)

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

    @property
    def ranking_table(self):
        rows = self._irt2_rows(self.ranking_results)
        table = tabulate(
            rows,
            headers=["", "both h@1", "both h@10", "both mrr"],
            floatfmt="2.3f",
        )

        return textwrap.indent("\n" + table, prefix="  ")


def projector(
    source: str,
    batch_size: Optional[int] = None,
    checkpoint: Optional[str] = None,
):
    print(irt2m.banner)
    log.info(f"run kgc evaluation for {source}")

    result = EvaluationResult(
        source,
        batch_size=batch_size,
        checkpoint=checkpoint,
    )

    print(result.kgc_table)
    print(result.ranking_table)


def _dic2row(dic, mapping):
    return {k: drslv(dic, v, default=0) for k, v in mapping.items()}


def _irt2row(dic, key, *contains):
    return {
        f"{key} {k}".replace("hits_at_", "h@"): v * 100
        for k, v in dflat(dic[key]).items()
        if all(c in k for c in contains)
    }


def _report_kgc_row(result):
    row = {}

    row |= _irt2row(result.kgc_results, "irt2_test", "micro")
    row |= _irt2row(result.kgc_results, "irt2_inductive", "micro")

    kgcrow = _dic2row(
        result.kgc_results,
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

    row |= {k: v * 100 for k, v in kgcrow.items()}

    row |= _dic2row(
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

    row |= _irt2row(result.kgc_results, "irt2_test", "macro")
    row |= _irt2row(result.kgc_results, "irt2_inductive", "macro")

    # supplemental

    row |= {
        "folder": str(result.data.path),
        "evaluated on": datetime.now().isoformat(),
    }

    return row


def _report_ranking_row(result):
    row = {}

    row |= _irt2row(result.ranking_results, "irt2_test", "micro")
    row |= _irt2row(result.ranking_results, "irt2_inductive", "micro")

    row |= _irt2row(result.ranking_results, "irt2_test", "macro")
    row |= _irt2row(result.ranking_results, "irt2_inductive", "macro")

    # supplemental

    row |= {
        "folder": str(result.data.path),
        "evaluated on": datetime.now().isoformat(),
    }


def create_report(folder: str):
    folder = kpath(folder, is_dir=True)
    glob = list(map(lambda p: p.parent, folder.glob("**/checkpoints")))

    print(f"loading {len(glob)} results")

    scenarios = dict(kgc=[], ranking=[])
    for source in glob:
        source = kpath(source, is_dir=True)
        result = EvaluationResult(source)

        print(textwrap.indent(str(result), prefix="  - "))

        scenarios["kgc"].append(_report_kgc_row(result))
        scenarios["ranking"].append(_report_kgc_row(result))

    for scenario, rows in scenarios.items():
        out = folder / f"{scenario}.summary.csv"
        with (out).open(mode="w") as fd:
            writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"wrote {out}")
