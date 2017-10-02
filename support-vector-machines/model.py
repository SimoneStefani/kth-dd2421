from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np, pylab as pl, random, math


# ================================================== #
# Kernel Functions
# ================================================== #

def linear_kernel(x, y):
    return np.dot(x, y) + 1

def poly_kernel(x, y, p=2):
    return np.power((np.dot(x, y) + 1), p)

def radial_kernel(x, y, sigma=2):
    diff = np.subtract(x, y)
    return math.exp((-np.dot(diff, diff)) / (2 * sigma * sigma))

def sigmoid_kernel(x, y, k=0.1, delta=0):
    np.transpose(x)
    return np.tanh(k * np.dot(x, y) - delta)


# ================================================== #
# Generate Points with Normal Distribution
# ================================================== #

def generate_data():
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(1.5, 1), 1.0) for i in range(10)] +\
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(10)]

    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

    return classA, classB


# ================================================== #
# Train SVM with Quadratic Optimization
# ================================================== #

def prepare_optimization(data, kernel, C):
    N = len(data)

    P = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            P[i][j] = (data[i])[2] * (data[j])[2] * kernel([(data[i])[0], (data[i])[1]], [(data[j])[0], (data[j])[1]])

    q = -np.ones((N, 1))
    h = np.zeros((N, 1))
    G = -np.eye(N)

    # Slack Variables
    if C != 0.:
        G = np.concatenate((np.eye(N), G))
        h = np.concatenate((C * np.ones((N, 1)), h))

    return P, q, h, G

def train(data, kernel, C=0):

    P, q, h, G = prepare_optimization(data, kernel, C)

    # Quadratic optimization with convex solver
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])

    # Filter non-zero vectors
    support_vectors=[]
    for i in range(0, len(alpha)):
        if alpha[i] > 1.e-5:
            support_vectors.append((alpha[i], (data[i])[0], (data[i])[1], (data[i])[2]))
    return support_vectors


# ================================================== #
# Plot Data and Boundaries
# ================================================== #

def plot_data(classA, classB):
    pl.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    pl.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')

def plot_boundaries(support_vectors, kernel):
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)

    grid = matrix([[indicator(support_vectors, x, y, kernel) for y in yrange] for x in xrange])
    
    pl.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

def indicator(svs, x, y, kernel):
    ind = 0.0

    for i in range(0, len(svs)):
        ind += (svs[i])[0] * (svs[i])[3] * kernel([x, y], [(svs[i])[1], (svs[i])[2]])

    return ind


# ================================================== #
# Main Classifier Function
# ================================================== #

def classify():
    random.seed(100)

    # Specify kernel:
    # - linear_kernel
    # - poly_kernel
    # - radial_kernel
    # - sigmoid_kernel
    kernel = poly_kernel

    classA, classB = generate_data()
    data = classA + classB
    random.shuffle(data)
    
    # Could pass C value for Slack variables
    # as third argument
    svs = train(data, kernel)    
    
    plot_data(classA, classB)
    plot_boundaries(svs, kernel)
    pl.show()

if __name__ == '__main__':
    classify()