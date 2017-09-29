from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np
import pylab, random, math


def linear_kernel(x, y):
    np.transpose(x)
    return np.dot(x, y) + 1

def poly_kernel(x, y, p=2):
	np.transpose(x)
	return math.pow((np.dot(x, y) + 1),p)

def radial_kernel(x, y, sigma=1):
	return math.exp((-np.dot(x - y, x - y))/(2 * math.pow(sigma, 2)))

def sigmoid_kernel(x, y, k=1, delta=1):
	np.transpose(x)
	return np.tanh(k * np.dot(x, y) - delta)