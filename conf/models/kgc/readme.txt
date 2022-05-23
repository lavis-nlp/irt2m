Structure is as follows:
  combine base.yaml with one of each [kind].*.yaml files

for example, to train ComplEx on CDE-Large using the LCWA training paradigm:

conf=conf/models/kgc
irt2m train kgc -c $conf/base.yaml -c $conf/models.complex.yaml -c $conf/training.lcwa.yaml -c $conf/data.cde-large.yaml
