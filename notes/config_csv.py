#!/usr/bin/env python
# creates the csv for the hyperparameter tables in the paper

import argparse
import csv
from collections import defaultdict

import yaml
from ktz.collections import dflat, drslv
from ktz.filesystem import path as kpath

# table layout
#
#   ranking|linking
#   joint|owe (single|multi context)
#   ----------------------------
#        | tiny | small | medium | large
#   opt  | ...

# expects summary.linking.csv and summary.ranking.csv
# produced by `irt2m evaluate create-report`


def RDic():
    return defaultdict(RDic)


def read_csvs(args):
    sources = dict(ranking=args.ranking, linking=args.linking)
    for name, path in sources.items():
        with kpath(path, is_file=True).open(mode="r", newline="") as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                yield name, row


MODEL_MAP = {
    "PSAC": ("OWE", "single"),
    "PMAC": ("OWE", "multi"),
    "JSC": ("JOINT", "single"),
    "JMC": ("JOINT", "multi"),
}

SPLIT_MAP = dict(T="tiny", S="small", M="medium", L="large")


OPTS = {
    "Embedding Dims.": "model_kwargs.embedding_dim",
    "Unfrozen Layer": "model_kwargs.freeze_except",
    "Regularizer Weight": "model_kwargs.regularizer_kwargs.weight",
    "Contexts per Sample": "module.train_ds_kwargs.contexts_per_sample",
    "Maximum Contexts": "module.train_ds_kwargs.max_contexts_per_sample",
    "Masked": "module.train_ds_kwargs.masking",
    "Batch Size": "module.train_loader_kwargs.batch_size",
    "Subbatch Size": "module.train_loader_kwargs.subbatch_size",
    "Learning Rate": "optimizer_kwargs.lr",
    "Weight Decay": "optimizer_kwargs.weight_decay",
    "Seed": "seed",
}

SEL_OPTS = {
    "JOINT": (
        "Embedding Dims.",
        "Unfrozen Layer",
        "Regularizer Weight",
        "Contexts per Sample",
        "Maximum Contexts",
        "Masked",
        "Batch Size",
        "Subbatch Size",
        "Learning Rate",
        "Weight Decay",
        "Seed",
    ),
    "OWE": (
        "Contexts per Sample",
        "Maximum Contexts",
        "Masked",
        "Batch Size",
        "Subbatch Size",
        "Learning Rate",
        "Seed",
    ),
}


def main(args):
    print(f"reading {args.ranking=} {args.linking=}\n")

    # data: ranking|linking -> model -> single|multi -> tiny|small|medium|large
    data = RDic()

    for name, row in read_csvs(args):
        prefix, suffix = row["prefix"].split("-")
        model, kind = MODEL_MAP[prefix]
        split = SPLIT_MAP[suffix]

        conf_path = kpath(kpath(row["folder"]) / "config.yaml", is_file=True)
        with conf_path.open(mode="r") as conf_fd:
            config = yaml.safe_load(conf_fd)

        print("found", name, model, kind, split)
        for opt, trail in OPTS.items():
            val = drslv(config, trail, sep=".", default=None)
            data[name][model][kind][split][opt] = val

    with kpath(args.out, is_file=False).open(mode="w", newline="") as fd:
        writer = csv.writer(fd)

        for trail, opts in dflat(data, only=3).items():
            fd.write(f"\n\n{trail}\n")
            _, model, _ = trail.split()

            splits = list(SPLIT_MAP.values())
            writer.writerow(["option"] + splits)

            for opt in SEL_OPTS[model]:
                row = [opt]
                for split in splits:
                    row.append(opts[split][opt])

                writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        required=True,
        help="target csv",
    )

    parser.add_argument(
        "--ranking",
        required=True,
        help="summary.ranking.csv",
    )
    parser.add_argument(
        "--linking",
        required=True,
        help="summary.linking.csv",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
