# -*- coding: utf-8 -*-
"""Train some neural models for IRT2."""

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pykeen as pk
import pykeen.pipeline
import pytorch_lightning as pl
import yaml
from irt2.dataset import IRT2
from ktz.collections import dmerge, ryaml
from ktz.filesystem import path as kpath
from pytorch_lightning.loggers import WandbLogger

from irt2m.data import Config, ProjectorModule, PyKEEN
from irt2m.models import PROJECTORS

log = logging.getLogger(__name__)


# --


def print_banner():
    print(
        """
              ┌──────────────────────────┐
              │ IRT2M PROJECTOR TRAINING │
              └──────────────────────────┘
        """
    )


def set_seed(config, seed: Optional[int] = None):
    if seed is not None:
        log.info("overwriting seed with cli arg")
        config["seed"] = seed

    if "seed" not in config:
        log.info("generating random seed")
        config["seed"] = random.randint(10**5, 10**7)

    log.info(f"! setting seed: {config['seed']}")
    pl.utilities.seed.seed_everything(config["seed"])


def join_kwargs(defaults: dict, **kwargs):
    return kwargs | defaults


# --


def _add_loghandler(out: str):
    # add an additional text logger
    log.info(f"adding an additional log handler: {out}")

    loghandler = logging.FileHandler(str(out), mode="w")
    loghandler.setLevel(log.getEffectiveLevel())
    loghandler.setFormatter(log.root.handlers[0].formatter)

    logging.getLogger("irt2m").addHandler(loghandler)


def _load_config_and_irt2(
    files: list[str],
    formatting: Callable[[Config], dict] = None,
) -> tuple[IRT2, Config]:
    timestamp = datetime.now()

    # as per our configuration wandb provides them as "-c foo"
    # which results in [" foo"] per config file entry
    config = ryaml(*map(str.strip, files))

    irt2 = IRT2.from_dir(path=config["irt2"])

    out = config["out"].format(
        dataset=irt2.config["create"]["name"].lower().replace("/", "-"),
        date=timestamp.strftime("%Y-%m-%d_%H-%M-%S"),
        **(formatting(config) if formatting else {}),
    )

    config["out"] = str(kpath(out, exists=False))
    config["timestamp"] = timestamp.isoformat()

    return irt2, config


# -- CW KGC TRAINING


def _kgc_handle_config(
    config: Config,
    irt2: IRT2,
    learning_rate: Optional[float] = None,
    embedding_dim: Optional[int] = None,
    regularizer_weight: Optional[float] = None,
    negatives: Optional[int] = None,
    loss: Optional[str] = None,
) -> Config:
    out = kpath(config["out"])

    defaults = {
        "pipeline": {
            "metadata": {"title": out.name},
            "random_seed": config["pykeen seed"],  # may be None
            "training_kwargs": {
                "checkpoint_directory": str(out / "checkpoints"),
                "checkpoint_name": config["pipeline"]["model"] + ".pt",
            },
        },
    }

    conf = dmerge(defaults, config)

    # some additional wandb configuration if used
    if conf["pipeline"]["result_tracker"] == "wandb":
        tracker_kwargs = conf["pipeline"].get("result_tracker_kwargs", {})
        tracker_kwargs = dmerge(
            {
                "tags": [
                    "kgc",
                    conf["pipeline"]["model"].lower(),
                    Path(conf["irt2"]).name.lower(),
                ],
                # "name": out.name,  - pykeen uses pipeline.title
                "dir": str(out),
            },
            tracker_kwargs,
        )

        # # finally copy whole configuration to wandb
        conf["pipeline"]["result_tracker_kwargs"] = tracker_kwargs

    # overwrite parameters

    def _overwrite(val, keys):
        if val is None:
            return

        target = conf
        for key in keys[:-1]:
            target[key] = target.get(key, {})
            target = target[key]

        log.info(f"overwriting {'.'.join(keys)} with {val}")
        target[keys[-1]] = val

    _overwrite(
        learning_rate,
        ["pipeline", "optimizer_kwargs", "lr"],
    )
    _overwrite(
        embedding_dim,
        ["pipeline", "model_kwargs", "embedding_dim"],
    )
    _overwrite(
        regularizer_weight,
        ["pipeline", "regularizer_kwargs", "weight"],
    )
    _overwrite(
        loss,
        ["pipeline", "loss"],
    )

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
    log.info("commence training of a closed-world KGC model")

    def formatting(config):
        return dict(model=config["pipeline"]["model"])

    irt2, conf = _load_config_and_irt2(config, formatting)
    conf = _kgc_handle_config(config=conf, irt2=irt2, **overwrites)

    logpath = kpath(conf["out"]) / "log.txt"
    _add_loghandler(logpath)
    log.info(f"writing additional logfile at {logpath}")

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
    pkds.to_path_binary(kpath(conf["out"]) / "triples")
    pk_results.save_to_directory((kpath(conf["out"]) / "pipeline").resolve())


# -- OW PROJECTOR TRAINING


def _init_logger(config):
    if "wandb" in config:
        logger = WandbLogger(
            project=config["wandb"]["project"],
            log_model=False,
        )

    else:
        log.warning("o wandb configuration found; falling back to csv")
        logger = pl.loggers.csv_logs.CSVLogger(config.out / "csv", name=name)

    # TODO: if resume is possible, allow_val_change if resuming
    logger.experiment.config.update(
        config,
        allow_val_change=False,
    )
    return logger


def projector(config: list[str]):
    """Train a projector."""

    def formatting(config):
        return dict(
            encoder=config["encoder"].lower(),
            projector=config["projector"].lower(),
        )

    irt2, config = _load_config_and_irt2(config, formatting)
    debug = config["trainer"]["fast_dev_run"]

    set_seed(config)

    logger = None if debug else _init_logger(config)
    datamodule = ProjectorModule(irt2, config)

    Projector = PROJECTORS[config["projector"]]
    model = Projector(irt2, config, datamodule.tokenizer)

    # configure

    out = kpath(config["out"], create=not debug)

    if debug:
        log.warning("running in debug mode")

    checkpoint_args = join_kwargs(
        config["checkpoint"],
        dirpath=out / "checkpoints",
    )

    callbacks = []
    if not debug:
        callbacks = [
            pl.callbacks.ModelCheckpoint(**checkpoint_args),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]

    if "early_stopping" in config:
        log.info("add early stopping callback")
        EarlyStopping = pl.callbacks.early_stopping.EarlyStopping
        callbacks.append(EarlyStopping(**config["early_stopping"]))

    trainer_args = join_kwargs(
        config["trainer"],
        logger=logger,
        callbacks=callbacks,
        weights_save_path=out / "weights",
    )

    trainer = pl.Trainer(**trainer_args)

    log.error("pre-training validation disabled")
    # trainer.validate(model, datamodule=datamodule)

    # config was augmented and rewritten with cached data
    # so the write needs to occur at the very end of initialization
    if not debug:
        with (out / "config.yml").open(mode="w") as fd:
            yaml.dump(config, fd)

    print_banner()
    trainer.fit(model, datamodule=datamodule)

    # TODO write results to disk
    # TODO try-except around fit to salvage data post-mortem
