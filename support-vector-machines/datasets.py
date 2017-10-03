import numpy as np, pylab as pl, random, math
import sklearn.datasets as dt


# ================================================== #
# Generate Points with Normal Distribution
# ================================================== #

def basic_dataset():
    # Not Linearly separable points
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] +\
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]

    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

    return classA, classB


# ================================================== #
# Generate Linearly Separable Points
# ================================================== #

def linearly_separable_dataset():
    # Linearly separable points
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(1.5, 1), 1.0) for i in range(10)] +\
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(10)]

    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

    return classA, classB


# ================================================== #
# Points in two interleaving half circles (without noise)
# ================================================== #

def pure_moons_dataset():

    X, Y = dt.make_moons(100)
    
    classC = []
    for i in range(len(X)):
        classC.append((X[i][0], X[i][1], 1 if Y[i] == 1 else -1))

    classA = [a for a in classC if (a)[2] == 1]
    classB = [a for a in classC if (a)[2] == -1]

    return classA, classB



# ================================================== #
# Points in two interleaving half circles (with noise)
# ================================================== #

def noisy_moons_dataset():

    X, Y = dt.make_moons(100, noise=0.2)
    
    classC = []
    for i in range(len(X)):
        classC.append((X[i][0], X[i][1], 1 if Y[i] == 1 else -1))

    classA = [a for a in classC if (a)[2] == 1]
    classB = [a for a in classC if (a)[2] == -1]

    return classA, classB


# ================================================== #
# Generate Points in Concentric Circles
# ================================================== #

def noisy_circles_dataset():
    # Make a large circle containing a smaller circle in 2d

    X, Y = dt.make_circles(100, factor=0.2, noise=0.1)
    classC = []
    for i in range(len(X)):
        classC.append((X[i][0], X[i][1], 1 if Y[i] == 1 else -1))

    classA = [a for a in classC if (a)[2] == 1]
    classB = [a for a in classC if (a)[2] == -1]

    return classA, classB