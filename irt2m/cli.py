# -*- coding: utf-8 -*-
"""Universal entry point for the cli."""

import logging
import os

import click
import pretty_errors

import irt2m
from irt2m import eval, train

irt2m.init_logging()


# ---

log = logging.getLogger(__name__)

os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"
# os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


@click.group()
def main():
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
def train_projector(**kwargs):
    """Train a projector that maps text to KGC vertices."""
    train.projector(**kwargs)


# --- EVALUATION


@main.group(name="evaluate")
def grp_evaluate():
    """Evaluating models."""
    pass


@grp_evaluate.command(name="projector")
@click.option(
    "--data",
    type=str,
    required=True,
    # multiple=True  # TODO run for all
    help="directory defined in config['out']",
)
@click.option(
    "--checkpoint",
    type=str,
    required=False,
    help="checkpoint name (e.g. last.ckpt); otherwise load best",
)
def evaluate_projector(**kwargs):
    """Evaluate a projector model"""
    eval.projector(**kwargs)
