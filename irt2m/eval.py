# -*- coding: utf-8 -*-
"""Evaluate models."""

import csv
import logging
import re
import textwrap
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path
from typing import Optional

import torch
import yaml
from irt2.dataset import IRT2
from ktz.collections import dmerge
from ktz.filesystem import path as kpath
from tabulate import tabulate
from tqdm import tqdm as _tqdm

import irt2m
from irt2m import data, models

log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=data.TERM_WIDTH)


def hits(dic, k):
    try:
        return dic["both"]["realistic"][f"hits_at_{k}"] * 100
    except KeyError:
        return 0.0


# TODO move classes like these to data?
class ProjectorResult:

    path: Path
    log: str
    config: data.Config
    validation: dict[dict[str, dict]]  # kind -> epoch -> results
    checkpoints: list[Path]

    @cached_property
    def description(self):
        header = str(self)
        checkpoints = "\n".join([f"  - {path.name}" for path in self.checkpoints])

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

        log.info(f"loaded {self}")

    # ---

    def load(self, checkpoint: str):
        log.info("loading checkpoint")

        print("loading IRT2...")
        irt2 = IRT2.from_dir(self.config["irt2"])
        print(f"loaded {irt2}")

        print("loading module...")
        module = data.ProjectorModule(irt2, self.config)

        Projector = models.MODELS[self.config["model"]]
        path = self.path / "checkpoints" / checkpoint
        model = Projector.load_from_checkpoint(
            path,
            irt2=irt2,
            config=self.config,
            tokenizer=module.tokenizer,
        )

        return irt2, module, model


def batcher(loader, module, device):
    for batch in loader:
        yield module.transfer_batch_to_device(batch, device, 0)


def tqdm_batcher(gen, loader, desc):
    yield from tqdm(
        gen,
        desc=desc,
        unit=" batches",
        total=len(loader),
    )


def create_projections(model, module, device):
    with torch.no_grad():
        model = model.to(device=device)
        model.clear_projections()  # paranoid

        # --

        print("\nCreate closed-world projections")
        loader = module.train_dataloader()

        breakpoint()

        gen = tqdm_batcher(
            gen=batcher(loader, module, device),
            loader=loader,
            desc="closed-world (train)",
        )

        for batch in gen:
            result = model.forward(batch)
            model.update_projections(*result)

        # --

        print("\nCreate open-world projections (validation)")
        loader = module.val_dataloader()

        gen = tqdm_batcher(
            gen=batcher(loader, module, device),
            loader=loader,
            desc="open-world (validation)",
        )

        for batch in gen:
            result = model.forward(batch)
            model.update_projections(*result)

        model.gather_projections()


def projector(
    source: str,
    batch_size: Optional[int] = None,
    checkpoint: Optional[str] = None,
):
    print(irt2m.banner)
    log.info(f"run evaluation for {source}")

    # load result

    source = kpath(source, is_dir=True)
    result = ProjectorResult(path=source)

    prefix = "  "
    print("Loaded:")
    print(textwrap.indent(str(result.description), prefix))
    print()

    assert checkpoint, "TODO load best checkpoint"

    # initialize model and data

    irt2, module, model = result.load(checkpoint)
    device = torch.device("cuda")

    # TODO cache projections

    # valid_ds: mention flat
    # valid_ds_kwargs:
    #   seed: 5012022
    #   max_contexts_per_sample: 1000
    # valid_loader_kwargs:
    #   shuffle: false
    #   batch_size: 30

    # create flat datasets to iterate all text contexts

    if batch_size is None:
        batch_size = result.config["module"]["valid_loader_kwargs"]["batch_size"]

    modconf = dict(
        module=dict(
            train_ds="mention flat",
            train_loader_kwargs=dict(batch_size=batch_size),
            valid_ds="mention flat",
            valid_loader_kwargs=dict(batch_size=batch_size),
        )
    )

    module = data.ProjectorModule(
        module.irt2,
        config=dmerge(result.config, modconf),
    )

    module.setup("evaluation")
    create_projections(model, module, device)

    breakpoint()

    # run evaluations

    # loader = (
    #     module.train_dataloader(),
    #     module.validation_dataloader(),
    #     # module.test_dataloader(),
    # )

    # for batch in batcher(loader, device=):
    #     projections, reduced_samples = self.forward(batch)
    #     self.update_projections()


def create_report(folder: str):
    folder = kpath(folder, is_dir=True)
    glob = list(map(lambda p: p.parent, folder.glob("**/checkpoints")))

    print(f"loading {len(glob)} results")

    def _rslv(dic, keychain):
        try:
            for key in keychain.split("."):
                dic = dic[key]
            return dic
        except KeyError:
            return None

    rows = []
    for source in glob:
        source = kpath(source, is_dir=True)
        result = ProjectorResult(path=source)

        print(textwrap.indent(str(result), prefix="  - "))

        row = {
            "prefix": "prefix",
            "date": "timestamp",
            "contexts per sample": "module.train_ds_kwargs.max_contexts_per_sample",
            "contexts per batch": "module.train_ds_kwargs.contexts_per_sample",
            "batch size": "module.train_loader_kwargs.batch_size",
            "subbatch size": "module.train_loader_kwargs.subbatch_size",
            "epochs": "trainer.max_epochs",
        }

        row = {k: _rslv(result.config, v) for k, v in row.items()}

        for k in 1, 10:
            for kind in "inductive", "transductive":
                hits, _, _ = result.best(kind, k=k)

                row |= {
                    # f"{kind} epoch": epoch,
                    f"{kind} h@{k}": hits,
                }

        row |= {"folder": str(source)}
        rows.append(row)

    out = folder / "summary.csv"
    with (out).open(mode="w") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {out}")
