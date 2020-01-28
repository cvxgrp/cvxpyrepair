import numpy as np
import cvxpy as cp
import warnings

import cvxpyrepair

np.random.seed(1515)

m = 5  # number of outcomes
n = 3  # number of wagers
R0 = np.random.randn(m, n)

x = cp.Variable(n)
t = cp.Variable(1)
R = cp.Parameter((m, n))
R.value = R0

prob = cp.Problem(cp.Maximize(cp.sum(R @ x)), [R @ x >= 0, cp.norm(x) <= 1, x >= 0])
prob.solve()
print(x.value)

prob = cp.Problem(cp.Maximize(cp.sum(R @ x)), [R @ x >= 0, x >= 0])
print(prob.solve())

np.set_printoptions(precision=2, suppress=True)


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)

print(bmatrix(R.value))


def r(params):
    R, = params
    return cp.norm((R - R0) / R0, 1), []

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cvxpyrepair.repair(prob, [R], r=r, verbose=True, lam=5)

print(bmatrix(R.value))
diff = R.value - R0
diff[np.abs(diff) < 1e-5] = 0.0
print(bmatrix(diff))

print(prob.solve())
