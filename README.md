# IRT2M - IRT2 Models

...

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
graph (`IRT2.closed_triples`). Subsequent, a projector is trained to
align input text descriptions with the trained KG-Embeddings.

First, train a closed-world KGC model:

```
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

```
conf=conf/models/kgc/
irt2m train kgc -c $conf/base.yaml -c $conf/models/complex-slcwa.yaml -c $conf/data/cde-large.yaml
```

It is also possible to run a hyperparameter search (using W&B sweeps):

```
wandb sweep conf/models/kgc/wandb/large-complex-slcwa-random.yaml
wandb agent URL
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
