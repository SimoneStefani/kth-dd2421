import monkdata as m
import dtree as dt
from prettytable import PrettyTable
import drawtree_qt5 as qt

monks = [m.monk1, m.monk2, m.monk3]

##---------------------------------------------------------
# COMPUTE ENTROPY OF DATASETS (ASS 1)
##---------------------------------------------------------
def compute_entropy():
  print ("Compute entropy of training datasets:")

  ent_table = PrettyTable(['Dataset', 'Entropy'])

  for i in range(3):
    l = ["MONK-{0}".format(i+1)]
    l.append(round(dt.entropy(monks[i]), 10))
    ent_table.add_row(l)

  print(ent_table)
  print ()


##---------------------------------------------------------
# COMPUTE INFORMATION GAIN (ASS 3)
##---------------------------------------------------------
def compute_gain():
  print ("Compute information gain of attributes in training datasets:")

  ig_table = PrettyTable(['Dataset', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])

  for i in range(3):
    l = ["MONK-{0}".format(i+1)]
    for k in range(6):
      l.append(round(dt.averageGain(monks[i], m.attributes[k]), 10))
    ig_table.add_row(l)

  print(ig_table)
  print ()


##---------------------------------------------------------
# BUILD DECISION TREES (ASS 5)
##---------------------------------------------------------
def compute_subsets(dataset, attribute):
  values = m.attributes[attribute].values
  subsets = []

  for val in values:
      subsets.append(dt.select(dataset, m.attributes[attribute], val))

  return subsets