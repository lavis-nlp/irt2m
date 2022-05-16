# -*- coding: utf-8 -*-
"""Universal entry point for the cli."""

import irt2m
from irt2m import train

import os
import logging
import textwrap

import click
import pretty_errors

irt2m.init_logging()


# ---

log = logging.getLogger(__name__)
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


def _create_banner(variables: dict[str, str]):
    variables = "\n".join(f"-  {k}: {v}" for k, v in variables.items())

    s = """
    --------------------
     IRT2M CLIENT
    --------------------

    variables in scope:
    {variables}
    """

    formatted = textwrap.dedent(s).format(variables=variables)
    return textwrap.indent(formatted, "  ")


@click.group()
def main():
    """Use irt2m from the command line."""
    log.info(" · IRT2M CLI ·")
    log.info(f"initialized root path: {irt2m.ENV.DIR.ROOT}")


@main.group(name="train")
def grp_train():
    """Training models."""
    pass


@grp_train.command(name="run")
def train_run():
    """Run a training."""
    train.run()
