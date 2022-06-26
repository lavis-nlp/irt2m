# -*- coding: utf-8 -*-
"""Evaluate models."""

import csv
import logging
import pickle
import re
import textwrap
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path
from typing import Optional

import torch
import torch.utils.data as td
import yaml
from irt2.dataset import IRT2
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

    dl_kwargs = config["module"]["valid_loader_kwargs"]
    dl_kwargs |= dict(shuffle=False, batch_size=batch_size)
    print(">> dataloader:", dl_kwargs)

    loaders = {
        "closed-world (train)": td.DataLoader(
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
        "open-world (validation)": td.DataLoader(
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
        "open-world (test)": td.DataLoader(
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
        model = model.to(device=device)

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


def init_projections(
    source: Path,
    config,
    irt2,
    model,
    module,
    device,
    batch_size,
):
    cache = kpath(source / "cache", create=True) / "projections.pkl"
    if cache.exists():
        print("loading projections from cache")

        with cache.open(mode="rb") as fd:
            P, PC = pickle.load(fd)

        model.projections = P
        model.projections_counts = PC

    else:
        print("cache miss! creating projections from text")

        P, PC = create_projections(
            config,
            irt2,
            model,
            module,
            device,
            batch_size,
        )

        with cache.open(mode="wb") as fd:
            pickle.dump((P, PC), fd)

    mask = PC != 0
    print(f"obtained {mask.sum().item()} projections: {P.shape}")


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
    model.eval()
    assert not model.training

    device = torch.device("cuda")
    if batch_size is None:
        batch_size = result.config["module"]["valid_loader_kwargs"]["batch_size"]

    # module.setup("evaluation")

    init_projections(
        source,
        result.config,
        irt2,
        model,
        module,
        device,
        batch_size,
    )

    # run evaluations

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
