from numpy import *
import numpy as np
import csv
from sklearn.cross_validation import KFold


def array_aux():
    x = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
    y = np.array([0, 1, 0, 1])
    z = np.array([('John', 0.), ('Sarah', 1.)])

    print x
    print x.shape
    print x.dtype
    print type(x)

    print z
    print z.dtype
    print z.ndim
    print

#array_aux()