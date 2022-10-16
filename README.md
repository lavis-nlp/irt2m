# IRT2M - IRT2 Models

This repository complements the [IRT2 (Inductive Reasoning with
Text)](https://github.com/lavis-nlp/irt2) repository and contains the
code used for training the baseline models presented in the paper.


## Installation

We recommend miniconda for python environment management. Python >= 3.9
is required. Package configuration can be found in the `setup.cfg`.

```
conda create irt2m --python=3.9
conda activate irt2m
pip install .
# or if you want to develop
pip install -e .[dev]
```

## OWE Training

This section details how to conduct the two-step training approach
outlined by [1, 2]. Here, a KGC model is trained on the closed-world
graph (`IRT2.closed_triples`). Subsequently, a projector is trained to
align input text descriptions with the trained KG-Embeddings.


### Closed-World KGC Model Training

First, train a closed-world KGC model:

```console
 $ irt2m train kgc --help
 Usage: irt2m train kgc [OPTIONS]

   Train a KGC model using PyKEEN.

   Options:
     -c, --config TEXT           configuration file  [required]
     --learning-rate FLOAT       optimizers' learning rate
     --embedding-dim INTEGER     embedding dimensionality
     --regularizer-weight FLOAT  L2-regularization
     --negatives                 negatives per positives (only sLCWA)
     --loss                      one of the PyKEEN loss functions
     --help                      Show this message and exit.
```

For example, training ComplEx [3] (we use [Weights & Biases](http://wandb.ai) as
the experiment logger - you can reconfigure this in conf/models/kgc/base.yaml):

```console
conf=conf/models/kgc/
irt2m train kgc -c $conf/base.yaml -c $conf/models/complex-slcwa.yaml -c $conf/data/cde-large.yaml
```

It is also possible to run a hyperparameter search (using W&B sweeps):

```console
wandb sweep conf/models/kgc/wandb/large-complex-slcwa-random.yaml
wandb agent URL
```

The main code entry point can be found in `irt2m.train.kgc(...)`.


### Open-World Projector Training

Secondly, a projector is trained. This is quite straight forward from
the CLI. To see the specific parameter configurations used in the
paper, please explore the files in `conf/models/projector`.

```console
$ irt2m train projector --help
Usage: irt2m train projector [OPTIONS]

  Train a projector that maps text to KGC vertices.

Options:
  -c, --config TEXT               configuration file  [required]
  --mode TEXT                     training mode
  --epochs INTEGER                how many epochs to train
  --contexts-per-sample INTEGER   overwrites contexts per sample for both
                                  training and validation
  --max-contexts-per-sample INTEGER
                                  overwrites max contexts per sample for both
                                  training and validation
  --masked BOOLEAN                whether to mask the mention in the text
                                  contexts
  --learning-rate FLOAT           overwrites learning rate
  --regularizer-weight FLOAT      overwrites embedding regularizer weight
  --weight-decay FLOAT            overwrites the optimizers' weight-decay
  --embedding-dim INTEGER         overwrites the models embedding dim (joint
                                  only)
  --freeze-except INTEGER         freeze all but the last N layers of the
                                  encoder
  --help                          Show this message and exit.
```

Please have a look at the configuration files in
`conf/models/projector/*yaml` for a detailed description of all
configuration options. The main code entry point can be found in
`irt2m.train.projector(...)`.

For example, training a single context projector on CDE-L:

```console
conf=conf/models/projector
irt2m train projector -c $conf/base.yaml -c $conf/data/irt2-cde-large.yaml
```

## Model Evaluation

To evaluate models trained with this code, simply invoke the
entry points for the respective tasks:

```console
$ irt2m evaluate linking --help
Usage: irt2m evaluate linking [OPTIONS]

  Evaluate a projector model for linking.

Options:
  --irt2 TEXT                irt2 dataset to load  [required]
  --source TEXT              directory defined in config['out']  [required]
  --checkpoint TEXT          checkpoint name (e.g. last.ckpt); otherwise load
                             best
  --batch-size INTEGER       batch-size, otherwise use validation batch size
  --irt2-batch-size INTEGER
  --help                     Show this message and exit.
```

```console
$ irt2m evaluate ranking --help
Usage: irt2m evaluate ranking [OPTIONS]

  Evaluate a projector model for ranking.

Options:
  --irt2 TEXT           irt2 dataset to load  [required]
  --source TEXT         directory defined in config['out']  [required]
  --checkpoint TEXT     checkpoint name (e.g. last.ckpt); otherwise load best
  --batch-size INTEGER  batch-size, otherwise use validation batch size
  --help                Show this message and exit.
```



# Bibliography

1. ```
   @inproceedings{shah2019open,
     title={An Open-World Extension to Knowledge Graph Completion Models},
     author={Shah, Haseeb and Villmow, Johannes and Ulges, Adrian and Schwanecke, Ulrich and Shafait, Faisal},
     booktitle={Thirty-Third AAAI Conference on Artificial Intelligence},
     year={2019}
   }
   ```
2. ```
   @inproceedings{hamann2021open,
     title={Open-World Knowledge Graph Completion Benchmarks for Knowledge Discovery},
     author={Hamann, Felix and Ulges, Adrian and Krechel, Dirk and Bergmann, Ralph},
     booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
     pages={252--264},
     year={2021},
     organization={Springer}
   }
   ```
3. ```
   @inproceedings{trouillon2016complex,
      title={Complex embeddings for simple link prediction},
      author={Trouillon, Th{\'e}o and Welbl, Johannes and Riedel, Sebastian and Gaussier, {\'E}ric and Bouchard, Guillaume},
      booktitle={International Conference on Machine Learning},
      pages={2071--2080},
      year={2016}
   }
   ```
