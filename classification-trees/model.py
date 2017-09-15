import monkdata as m
import dtree as dt


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

