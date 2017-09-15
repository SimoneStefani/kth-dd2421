import monkdata as m
import dtree as dt
from prettytable import PrettyTable


##---------------------------------------------------------
# COMPUTE ENTROPY OF DATASETS (ASS 1)
##---------------------------------------------------------
ent_monk1 = dt.entropy(m.monk1)
ent_monk2 = dt.entropy(m.monk2)
ent_monk3 = dt.entropy(m.monk3)

print ("Compute entropy of training datasets:")
print ("- Entropy of monk1: ", ent_monk1)
print ("- Entropy of monk2: ", ent_monk2)
print ("- Entropy of monk3: ", ent_monk3)
print ()


##---------------------------------------------------------
# COMPUTE INFORMATION GAIN (ASS 3)
##---------------------------------------------------------
print ("Compute information gain of attributes in training datasets:")
ig_table = PrettyTable(['Dataset', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])

for i in range(3):
  l = ["MONK-{0}".format(i+1)]
  for k in range(6):
    gain = dt.averageGain(getattr(m, "monk{0}".format(i+1)), m.attributes[k])
    l.append(round(gain, 10))
  ig_table.add_row(l)

print(ig_table)



