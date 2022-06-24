# -*- coding: utf-8 -*-
"""Train some neural models for IRT2."""

import logging
import random
import sys
import traceback
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
from pytorch_lightning.callbacks.progress import tqdm_progress
from pytorch_lightning.loggers import WandbLogger

import irt2m
from irt2m import data
from irt2m.models import MODELS

log = logging.getLogger(__name__)


# --

# used for logging
REG = {
    "single context affine projector": dict(
        prefix="PSA",
        tags=["single", "projector"],
    ),
    "multi context affine projector": dict(
        prefix="PMA",
        tags=["multi", "projector"],
    ),
}

assert set(REG) == set(MODELS), "missing registry information"


# --


def set_seed(config, seed: Optional[int] = None):
    if seed is not None:
        log.info("overwriting seed with cli arg")
        config["seed"] = seed

    if "seed" not in config:
        log.info("generating random seed")
        config["seed"] = random.randint(10**5, 10**7)

    log.info(f"! setting seed: {config['seed']}")
    pl.seed_everything(config["seed"], workers=True)


def join_kwargs(defaults: dict, **kwargs):
    return kwargs | defaults


# --


def _add_loghandler(out: str):
    # add an additional text logger

    loghandler = logging.FileHandler(str(out), mode="w")
    loghandler.setLevel(log.getEffectiveLevel())
    loghandler.setFormatter(log.root.handlers[0].formatter)

    logging.getLogger("irt2m").addHandler(loghandler)


def _load_config_and_irt2(
    files: list[str],
    handler: Callable[[data.Config], dict] = None,
) -> tuple[IRT2, data.Config]:
    timestamp = datetime.now()

    # as per our configuration wandb provides them as "-c foo"
    # which results in [" foo"] per config file entry
    config = ryaml(*map(str.strip, files))

    irt2 = IRT2.from_dir(path=config["irt2"])

    out = config["out"].format(
        dataset=irt2.config["create"]["name"].lower().replace("/", "-"),
        date=timestamp.strftime("%Y-%m-%d_%H-%M-%S"),
        **(handler(config, irt2) if handler else {}),
    )

    config["out"] = str(kpath(out, exists=False))
    config["timestamp"] = timestamp.isoformat()

    return irt2, config


# -- CW KGC TRAINING


def _kgc_handle_config(
    config: data.Config,
    irt2: IRT2,
    learning_rate: Optional[float] = None,
    embedding_dim: Optional[int] = None,
    regularizer_weight: Optional[float] = None,
    negatives: Optional[int] = None,
    loss: Optional[str] = None,
) -> data.Config:
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

    def config_handler(config, irt2):
        return dict(model=config["pipeline"]["model"])

    irt2, conf = _load_config_and_irt2(config, config_handler)
    conf = _kgc_handle_config(config=conf, irt2=irt2, **overwrites)

    logpath = kpath(conf["out"]) / "log.txt"
    _add_loghandler(logpath)
    log.info(f"writing additional logfile at {logpath}")

    pkds = data.PyKEEN.from_irt2(
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


# -- SHARED PYTORCH LIGHTNING


def _fit(trainer, model, datamodule):
    trainer.fit(model, datamodule=datamodule)


def fit(trainer, model, datamodule, debug):

    if debug:
        _fit(trainer, model, datamodule)
        return

    out = kpath(model.config["out"], is_dir=True)

    try:
        _fit(trainer, model, datamodule)

    except Exception as exc:
        out_err = kpath(out / "death", create=True)

        log.error(f"an exception occured: >[{exc}]<")
        log.error(f"write traces to {out_err}")

        _, _, trace = sys.exc_info()
        callstack = "\n".join(traceback.format_tb(trace))

        # write stacktrace

        with (out_err / "stacktrace.txt").open(mode="w") as fd:
            fd.write(callstack)

        # write model checkpoint

        if model is not None:
            log.error("write model checkpoint")
            trainer.save_checkpoint(str(out_err / "tombstone.ckpt"))

        print(f"Exception: {exc}")
        print(callstack)


# -- OW PROJECTOR TRAINING


# registers config['prefix']
def _config_add_prefix(config, irt2) -> str:
    prefix = REG[config["model"]]["prefix"]

    if "kgc" in config:
        kgc_model = list(config["kgc"])[0]
        if kgc_model == "complex":
            prefix += "C"

    data = irt2.name[-1]
    config["prefix"] = f"{prefix}-{data}"


# registers config['tags']
def _config_add_tags(config, irt2) -> str:
    tags = [
        irt2.name,
        config["encoder"],
        config["module"]["train_ds"],
    ]

    if config["model"].endswith("projector"):
        tags += REG[config["model"]]["tags"]

    if "kgc" in config:
        tags += list(config["kgc"])

    config["tags"] = tags


def _init_logger(config, debug):

    if "wandb" in config:
        path = kpath(config["out"], create=True)

        kwargs = dict(
            name=f"[{config['prefix']}] {config['timestamp']}",
            project=config["wandb"]["project"],
            log_model=False,
            save_dir=str(path.resolve()),
            tags=config["tags"],
            mode="offline" if debug else "online",
        )

        logger = WandbLogger(**kwargs)

    # register other logger here if necessary

    # TODO: if resume is possible, allow_val_change if resuming
    logger.experiment.config.update(
        config,
        allow_val_change=False,
    )

    return logger


class ProgressCallback(tqdm_progress.TQDMProgressBar):
    # unfortunately, there is no way to pass up additional kwargs
    # to the different super().init_validation_tqdm() calls

    shared_kwargs = dict(
        leave=False,
        ncols=data.TERM_WIDTH,
        file=sys.stdout,
    )

    def init_sanity_tqdm(self):
        return tqdm_progress.Tqdm(
            desc=self.sanity_check_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            **ProgressCallback.shared_kwargs,
        )

    def init_train_tqdm(self):
        return tqdm_progress.Tqdm(
            desc=self.train_description,
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            **ProgressCallback.shared_kwargs,
        )

    def init_predict_tqdm(self):
        return tqdm_progress.Tqdm(
            desc=self.predict_description,
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            **ProgressCallback.shared_kwargs,
        )

    def init_validation_tqdm(self):
        has_main_bar = self.trainer.state.fn != "validate"

        return tqdm_progress.Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            **ProgressCallback.shared_kwargs,
        )

    def init_test_tqdm(self):
        return tqdm_progress.Tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            **ProgressCallback.shared_kwargs,
        )


def _init_callbacks(config, debug):
    callbacks = [ProgressCallback()]

    if debug:
        return callbacks

    checkpoint_args = join_kwargs(
        config["checkpoint"],
        dirpath=(kpath(config["out"]) / "checkpoints").resolve(),
    )

    callbacks += [
        pl.callbacks.ModelCheckpoint(**checkpoint_args),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if "early_stopping" in config:
        log.info("add early stopping callback")
        EarlyStopping = pl.callbacks.early_stopping.EarlyStopping
        callbacks.append(EarlyStopping(**config["early_stopping"]))

    return callbacks


def projector(config: list[str]):
    """Train a projector."""

    def config_handler(config, irt2):
        _config_add_prefix(config, irt2)
        _config_add_tags(config, irt2)

        return dict(
            encoder=config["encoder"].lower(),
            projector=config["model"].lower(),
            prefix=config["prefix"],
        )

    irt2, config = _load_config_and_irt2(config, config_handler)
    debug = config["trainer"]["fast_dev_run"]
    out = kpath(config["out"], create=not debug)

    if not debug:
        _add_loghandler(out / "log.txt")

    if debug:
        log.warning("running in debug mode")

    set_seed(config)

    # initialize

    datamodule = data.ProjectorModule(irt2, config)
    Projector = MODELS[config["model"]]
    model = Projector(irt2, config, datamodule.tokenizer)
    logger = None if debug else _init_logger(config, debug)

    # configure

    callbacks = _init_callbacks(config, debug)

    trainer_kwargs = join_kwargs(
        config["trainer"],
        logger=logger,
        callbacks=callbacks,
        weights_save_path=out / "weights",
        deterministic=True,
    )

    trainer = pl.Trainer(**trainer_kwargs)

    # config was augmented and rewritten with cached data
    # so the write needs to occur at the very end of initialization
    if not debug:
        with (out / "config.yaml").open(mode="w") as fd:
            yaml.dump(config, fd)

    print("\n" + irt2m.banner + "\n")
    log.info("--- rise, if you would")

    fit(trainer, model, datamodule, debug)
