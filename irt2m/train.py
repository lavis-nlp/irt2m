# -*- coding: utf-8 -*-
"""Train some neural models for irt2."""

from irt2m.data import PyKEEN

from irt2.dataset import IRT2

import yaml

import pykeen as pk
import pykeen.pipeline

from ktz.collections import ryaml
from ktz.collections import dmerge
from ktz.filesystem import path as kpath

import logging
from pathlib import Path
from datetime import datetime

from typing import Optional

log = logging.getLogger(__name__)


# -- CW KGC TRAINING


def _kgc_handle_config(
    conf: dict,
    irt2: IRT2,
    learning_rate: Optional[float] = None,
    embedding_dim: Optional[int] = None,
    regularizer_weight: Optional[float] = None,
    negatives: Optional[int] = None,
    loss: Optional[str] = None,
):
    timestamp = datetime.now()

    out = kpath(
        conf["out"].format(
            dataset=irt2.config["create"]["name"].lower().replace("/", "-"),
            date=timestamp.strftime("%Y-%m-%d_%H-%M-%S"),
            model=conf["pipeline"]["model"].lower(),
        ),
        exists=False,
    )

    # augment configuration

    defaults = {
        "created": timestamp.isoformat(),
        "pipeline": {
            "metadata": {"title": out.name},
            "random_seed": conf["pykeen seed"],  # may be None
            "training_kwargs": {
                "checkpoint_directory": str(out / "checkpoints"),
                "checkpoint_name": conf["pipeline"]["model"] + ".pt",
            },
        },
    }

    overwrites = {
        "out": str(out),
    }

    conf = dmerge(defaults, conf)
    conf = dmerge(conf, overwrites)

    # some additional wandb configuration if used
    if conf["pipeline"]["result_tracker"] == "wandb":
        tracker_kwargs = conf["pipeline"].get("result_tracker_kwargs", {})
        tracker_kwargs = dmerge(
            {
                "tags": [
                    "kgc",
                    conf["pipeline"]["model"].lower(),
                    Path(conf["dataset"]).name.lower(),
                ],
                # "name": out.name,  - pykeen uses pipeline.title
                "dir": str(out),
            },
            tracker_kwargs,
        )

        # # finally copy whole configuration to wandb
        conf["pipeline"]["result_tracker_kwargs"] = tracker_kwargs

    # overwrite parameters

    def _overwrite(conf, val, *keys):
        if val is None:
            return

        target = conf
        for key in keys[:-1]:
            target[key] = target.get(key, {})
            target = target[key]

        log.info(f"overwriting {'.'.join(keys)} with {val}")
        target[keys[-1]] = val

    _overwrite(conf, learning_rate, "pipeline", "optimizer_kwargs", "lr")
    _overwrite(conf, embedding_dim, "pipeline", "model_kwargs", "embedding_dim")
    _overwrite(conf, regularizer_weight, "pipeline", "regularizer_kwargs", "weight")
    _overwrite(conf, loss, "pipeline", "loss")

    # write config

    log.info(f"saving config to {out}")
    out.mkdir(exist_ok=True, parents=True)
    with (out / "config.yaml").open(mode="w") as fd:
        yaml.safe_dump(conf, fd)

    return conf


def kgc(
    config: list[str],
    **overwrites,
):
    """Train a KGC model."""
    assert config is not None

    # as per our configuration wandb provides them as "-c foo"
    # which results in [" foo"] per config file entry
    config = map(str.strip, config)

    log.info("commence training of a closed-world KGC model")
    log.info(config)

    # configuration files and command line args
    conf = ryaml(*config)

    irt2 = IRT2.from_dir(path=conf["dataset"])
    conf = _kgc_handle_config(conf=conf, irt2=irt2, **overwrites)

    pkds = PyKEEN.from_irt2(
        dataset=irt2,
        ratio=conf["ratio"],
        seed=conf["dataset seed"],
    )

    print("\nRunning KGC training with", conf["pipeline"]["model"])
    print("Dataset:", str(pkds))

    pk_results = pk.pipeline.pipeline(
        # data
        training=pkds.training,
        validation=pkds.validation,
        testing=pkds.validation,
        # yaml options
        **conf["pipeline"],
    )

    # pykeen calls .as_uri on the path -> .resolve() required
    pk_results.save_to_directory((kpath(conf["out"]) / "pykeen").resolve())
