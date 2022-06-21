# -*- coding: utf-8 -*-
"""Evaluate models."""

import logging
import re
import textwrap
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Optional

import yaml
from irt2.dataset import IRT2
from ktz.filesystem import path as kpath
from tabulate import tabulate

import irt2m
from irt2m import data, models

log = logging.getLogger(__name__)


def _get_h10(dic):
    try:
        return dic['both']['realistic']['hits_at_10']
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
            row = tuple(_get_h10(self.validation[kind][key]) for kind in tablehead)
            tablerows.append(key + row)

        tablerows.sort(key=lambda t: t[0])  # asc by epoch
        tablehead = ("epoch", "step") + tablehead

        return "\n".join(
            [
                header,
                "\nCheckpoints:",
                checkpoints,
                "\nValidation:",
                tabulate(tablerows, headers=tablehead),
            ]
        )

    def __str__(self):
        timestamp = datetime.fromisoformat(self.config["timestamp"])
        prefix = self.config["prefix"]

        vals = self.validation["inductive"]
        gen = ((_get_h10(dic), epoch) for (epoch, _), dic in vals.items())
        best = sorted(gen, reverse=True)

        h10, epoch = best[0]
        max_epoch = max(epoch for _, epoch in best)

        return (
            f"Projectors {prefix} {timestamp}"
            f" (best h@10: {h10:2.3f} at epoch {epoch}/{max_epoch})"
        )

    def __init__(self, path: Path):
        log.info(f"initializing trained projectors from {path}")
        self.path = path

        with (path / "log.txt").open(mode="r") as fd:
            self.log = fd.read()

        with (path / "config.yaml").open(mode="r") as fd:
            self.config = yaml.safe_load(fd)

        self.validation = {"train": {}, "transductive": {}, "inductive": {}}
        for glob in path.glob("kgc/*yaml"):
            # expecting file names like epoch=0-step=0_inductive.yaml
            _, epoch, _, step, kind = re.split(r"[=\-_]", glob.stem)
            with glob.open(mode="r") as fd:
                key = int(epoch), int(step)
                self.validation[kind][key] = yaml.safe_load(fd)

        self.checkpoints = []
        for glob in path.glob("checkpoints/*ckpt"):
            self.checkpoints.append(glob)

        log.info("loaded {self}")

    # ---

    def load(self, checkpoint: str):
        log.info("loading checkpoint")

        print("loading IRT2...")
        irt2 = IRT2.from_dir(self.config["irt2"])
        print(f"loaded {irt2}")

        print("loading module...")
        module = data.ProjectorModule(irt2, self.config)

        Projector = models.MODELS[self.config["model"]]
        path = self.path / 'checkpoints'/ checkpoint
        model = Projector.load_from_checkpoint(
            path,
            irt2=irt2,
            config=self.config,
            tokenizer=module.tokenizer,
        )

        return irt2, module, model


# def batcher(loader, device)
#     with torch.no_grad():
#         model.clear_projections()  # paranoid
#         for loader in loaders:
#             for batch in loader:
#                 yield module.transfer_batch_to_device(batch, device, 0)


def projector(data: str, checkpoint: Optional[str] = None):
    print(irt2m.banner)
    log.info(f"run evaluation for {data}")

    data = kpath(data, is_dir=True)
    result = ProjectorResult(path=data)

    prefix = "  "
    print("Loaded:")
    print(textwrap.indent(str(result.description), prefix))
    print()

    assert checkpoint, "TODO load best checkpoint"
    irt2, module, model = result.load(checkpoint)

    # loader = (
    #     module.train_dataloader(),
    #     module.validation_dataloader(),
    #     # module.test_dataloader(),
    # )

    # for batch in batcher(loader, device=):
    #     projections, reduced_samples = self.forward(batch)
    #     self.update_projections()
