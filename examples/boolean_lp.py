import numpy as np
import cvxpy as cp
import warnings

import cvxpyrepair

# generate problem
seed = 125  # ee 125 anniversary
np.random.seed(seed)

n = 6
m = 3

A = np.random.randn(m, n)
b = A @ np.random.randint(0, 2, size=n)
c = np.random.randn(n)

# solve using cvxpy
x = cp.Variable(n, boolean=True)

prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b])
result = prob.solve()

x_opt = x.value
print(x_opt)

# solve by finding the closest feasible problem
theta = cp.Parameter(n)
theta.value = .5 * np.ones(n)
x = cp.Variable(n)
objective = cp.Minimize(0.0)
constraints = [x == theta, cp.multiply(theta, x) == x, A @ x <= b]
prob = cp.Problem(objective, constraints)


def r(params):
    theta, = params
    return c @ theta, [theta >= 0, theta <= 1]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cvxpyrepair.repair(prob, [theta], r=r, verbose=True, lam=1, lr=.1)

print(theta.value)
assert np.linalg.norm(theta.value - x_opt) <= 1e-3
