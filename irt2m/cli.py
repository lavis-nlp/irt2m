# -*- coding: utf-8 -*-
"""Universal entry point for the cli."""

import logging
import os

import click
import pretty_errors
import torch
from pudb.var_view import default_stringifier as pudb_str

import irt2m
from irt2m import evaluation, train

# ---


log = logging.getLogger(__name__)

# os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"


# To better render tensors and to avoid cluttering
# the logfile with len() of zero-length tensor exceptions
# Register with Ctl-P -> Variable Stringifier -> Custom
# and enter the path to this file
def pudb_stringifier(obj):
    if isinstance(obj, torch.Tensor):
        dtype = str(obj.dtype).split(".")[1]

        if not obj.numel():
            suffix = "empty"
        elif obj.numel() == 1:
            suffix = f"value: {obj.item()}"
        else:
            suffix = "dims: " + " x ".join(map(str, obj.shape))

        return f"Tensor({dtype}) {suffix}"

    return pudb_str(obj)


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


# ---


@click.group()
def main():
    irt2m.init_logging()

    """Use irt2m from the command line."""
    log.info(f"\n\n{irt2m.banner}\n")
    log.info(f"initialized root path: {irt2m.ENV.DIR.ROOT}")
    log.info(f"executing from: {os.getcwd()}")


@main.group(name="train")
def grp_train():
    """Training models."""
    pass


# --- KGC TRAINING


@grp_train.command(name="kgc")
@click.option(
    "-c",
    "--config",
    type=str,
    multiple=True,
    required=True,
    help="configuration file",
)
@click.option(
    "--learning-rate",
    type=float,
    required=False,
    help="optimizers' learning rate",
)
@click.option(
    "--embedding-dim",
    type=int,
    required=False,
    help="embedding dimensionality",
)
@click.option(
    "--regularizer-weight",
    type=float,
    required=False,
    help="L2-regularization",
)
@click.option(
    "--negatives",
    type=int,
    help="negatives per positives (only sLCWA)",
)
@click.option(
    "--loss",
    type=str,
    help="one of the PyKEEN loss functions",
)
def train_kgc(**kwargs):
    """Train a KGC model using PyKEEN."""
    train.kgc(**kwargs)


# --- PROJECTOR TRAINING


@grp_train.command(name="projector")
@click.option(
    "-c",
    "--config",
    type=str,
    multiple=True,
    required=True,
    help="configuration file",
)
# config overwrites
@click.option(
    "--mode",
    type=str,
    required=False,
    help="training mode",
)
@click.option(
    "--epochs",
    type=int,
    multiple=False,
    required=False,
    help="how many epochs to train",
)
@click.option(
    "--contexts-per-sample",
    type=int,
    multiple=False,
    required=False,
    help="overwrites contexts per sample for both training and validation",
)
@click.option(
    "--max-contexts-per-sample",
    type=int,
    multiple=False,
    required=False,
    help="overwrites max contexts per sample for both training and validation",
)
@click.option(
    "--masked",
    type=bool,  # cannot use is_flag because of wandb sweeps
    default=None,
    required=False,
    help="whether to mask the mention in the text contexts",
)
@click.option(
    "--learning-rate",
    type=float,
    default=None,
    required=False,
    help="overwrites learning rate",
)
@click.option(
    "--regularizer-weight",
    type=float,
    default=None,
    required=False,
    help="overwrites embedding regularizer weight",
)
@click.option(
    "--weight-decay",
    type=float,
    default=None,
    required=False,
    help="overwrites the optimizers' weight-decay",
)
@click.option(
    "--embedding-dim",
    type=int,
    default=None,
    required=False,
    help="overwrites the models embedding dim (joint only)",
)
@click.option(
    "--freeze-except",
    type=int,
    default=None,
    required=False,
    help="freeze all but the last N layers of the encoder",
)
def train_projector(**kwargs):
    """Train a projector that maps text to KGC vertices."""
    train.projector(**kwargs)


# --- EVALUATION


@main.group(name="evaluate")
def grp_evaluate():
    """Evaluate models."""
    pass


@grp_evaluate.command(name="linking")
@click.option(
    "--irt2",
    required=True,
    help="irt2 dataset to load",
)
@click.option(
    "--source",
    type=str,
    required=True,
    help="directory defined in config['out']",
)
@click.option(
    "--checkpoint",
    type=str,
    required=False,
    help="checkpoint name (e.g. last.ckpt); otherwise load best",
)
@click.option(
    "--batch-size",
    type=int,
    required=False,
    help="batch-size, otherwise use validation batch size",
)
def evaluate_linking(**kwargs):
    """Evaluate a projector model for linking."""
    evaluation.linking(**kwargs)


@grp_evaluate.command(name="ranking")
@click.option(
    "--irt2",
    required=True,
    help="irt2 dataset to load",
)
@click.option(
    "--source",
    type=str,
    required=True,
    help="directory defined in config['out']",
)
@click.option(
    "--checkpoint",
    type=str,
    required=False,
    help="checkpoint name (e.g. last.ckpt); otherwise load best",
)
@click.option(
    "--batch-size",
    type=int,
    required=False,
    help="batch-size, otherwise use validation batch size",
)
def evaluate_ranking(**kwargs):
    """Evaluate a projector model for ranking."""
    evaluation.ranking(**kwargs)


@grp_evaluate.command(name="create-report")
@click.option(
    "--folder",
    type=str,
    required=True,
    help="directory where **/checkpoints/ can be found",
)
def create_evaluation_report(**kwargs):
    """Evaluate a projector model"""
    evaluation.create_report(**kwargs)


# ---

# we need to actively pull all other modules
# which register click commands

import irt2m.migrations  # noqa

# which register click commands
