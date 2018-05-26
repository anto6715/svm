from scipy.optimize import minimize
import scipy

import sys

fun = lambda x: x[0]
print(callable(fun))
res =  minimize(fun, (0,0))
print(res.x)