import itertools
import math

def iden(x):
    return x

def log(x):
    return math.log(x)

def exp(x):
    return 1/(1+math.exp(-x))

X = [1, 2, 3, 4]
funcs = [iden, log, exp]

def apply(functions, values):
    return [func(val) for func,val in zip(functions, values)]

values = [apply(f, X) for f in itertools.product(funcs, repeat=len(X))]
print(len(values))