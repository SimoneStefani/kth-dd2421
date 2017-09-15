from prettytable import PrettyTable
import drawtree_qt5 as qt
import monkdata as m
import random as rnd
import dtree as dt

monks = [m.monk1, m.monk2, m.monk3]
monks_test = [m.monk1test, m.monk2test, m.monk3test]

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
def compute_trees_errors():
  print ("Compute the train and test set errors for the full trees:")

  err_table = PrettyTable(['Dataset', 'Error (train)', 'Error (test)'])

  for i in range(3):
    l = ["MONK-{0}".format(i+1)]

    t = dt.buildTree(monks[i], m.attributes)
    l.append(1 - dt.check(t, monks[i]))
    l.append(1 - dt.check(t, monks_test[i]))

    err_table.add_row(l)

  print(err_table)
  print ()


##---------------------------------------------------------
# PRUNE TREES (ASS 7)
##---------------------------------------------------------
def partition(data, fraction):
  ldata = list(data)
  rnd.shuffle(ldata)
  breakPoint = int(len(ldata) * fraction)
  return ldata[:breakPoint], ldata[breakPoint:]

def prune_trees(data, test):
  fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  pruned = []

  for fraction in fractions:
    train, validate = partition(data, fraction)
    tree = dt.buildTree(train, m.attributes)
    forest = dt.allPruned(tree)
    best_perf = dt.check(tree, validate)

    temp_tree = 0
    best_tree = tree

    for t in forest:
      temp_perf = dt.check(t, validate)
      if best_perf < temp_perf:
        best_perf = temp_perf
        best_tree = t

    pruned.append((1 - check(best_tree, test), fraction))

  return pruned

def evaluate_pruning():
  fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  monk1_pruned = []
  monk3_pruned = []

  for i in range(100):
    monk1_pruned.append(prune_trees(m.monk1, m.monk1test))
    monk3_pruned.append(prune_trees(m.monk3, m.monk3test))

  



  