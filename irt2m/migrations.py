# -*- coding: utf-8 -*-

from ktz.filesystem import path as kpath
import torch
import click


from irt2m.cli import main


@main.group(name="migrate")
def grp_migrate():
    """Run migrations."""
    pass


# --


@grp_migrate.command(name="220706-jsc-ckpt")
@click.option(
    "--folder",
    type=str,
    required=True,
    help="directory where checkpoints can be found (**/*.ckpt)",
)
def _220706_jsc_ckpt(folder: str):
    """
    Adjust state dict of checkpoints for legacy JSC and JMC models.

    2022/06/07 - Migrate state dicts for JSC models
    They no longer have a ff layer called "project"
    but use the parent class' forward() function and
    a ff layer called "projector".

    For all checkpoints of jsc models, the state dict
    must be changed such that project.bias and project.weight
    are renamed to projector.*

    The function accepts a folder which is recursively walked
    to find all *ckpt files.
    """

    folder = kpath(folder, is_dir=True)
    for path in folder.glob('**/*.ckpt'):

        print(f'\nlooking at {path.parent.parent.name}: {path.name}')

        print('loading state dict...')
        state = torch.load(path)
        state_dict = state['state_dict']

        if 'project.weight' not in state_dict:
            print('already fixed: skipping')
            continue

        print('making a copy...')
        torch.save(state, path.parent / (path.name + '.bak'))

        print('fixing state dict')
        state_dict['projector.weight'] = state_dict['project.weight']
        del state_dict['project.weight']

        state_dict['projector.bias'] = state_dict['project.bias']
        del state_dict['project.bias']

        print('overwriting original...')
        torch.save(state, path)
