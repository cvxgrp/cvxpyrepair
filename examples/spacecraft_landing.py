import numpy as np
import cvxpy as cp

import cvxpyrepair
import warnings

np.random.seed(1)

m_0 = 12.
max_fuel_0 = 200.
max_thrust_0 = 50.
alpha_0 = .5

m = cp.Parameter(1, nonneg=True)
max_fuel = cp.Parameter(1, nonneg=True)
max_thrust = cp.Parameter(1, nonneg=True)
alpha = cp.Parameter(1, nonneg=True)

m.value = np.array([m_0])
max_fuel.value = np.array([max_fuel_0])
max_thrust.value = np.array([max_thrust_0])
alpha.value = np.array([alpha_0])

T_final = 10
h = 1
g = 9.8
gamma = 1.

H = int(T_final / h)

e_down = -np.array([0, 0, 1])
initial_position = np.array([10, 10, 50])
initial_velocity = np.array([10, -10, -10])

x = cp.Variable((3, H))  # position
v = cp.Variable((3, H))  # velocity
f = cp.Variable((3, H - 1))  # force

cons = [
    x[:, 0] == initial_position,
    v[:, 0] == initial_velocity,
    x[:, -1] == 0,
    v[:, -1] == 0,
    f[-1, :] >= alpha * cp.norm(f[:2, :], axis=0),

    m[0] * v[:, 1:] == m[0] * v[:, :-1] + h * f + h * g *
    np.tile(e_down[:, np.newaxis], (1, H - 1)),
    x[:, 1:] == x[:, :-1] + h * (v[:, :-1] + v[:, 1:]) / 2,
    cp.norm(f, axis=0) <= max_thrust[0],
    h * gamma * sum(cp.norm(f, axis=0)) <= max_fuel[0]
]

obj = cp.Minimize(cp.norm(v, 'fro') / H)
prob = cp.Problem(obj, cons)

print(prob.solve())


def penalty_function(param_list):
    m, max_fuel, alpha, max_thrust = param_list

    return (cp.abs(m - m_0) / m_0
            + cp.abs(max_thrust - max_thrust_0) / max_thrust_0
            + cp.abs(alpha - alpha_0) / alpha_0
            + cp.abs(max_fuel - max_fuel_0) / max_fuel_0), [m >= 9, alpha >= 0.2, max_fuel >= 0.1,
                                                            max_thrust >= 0.1]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cvxpyrepair.repair(prob, [
                       m, max_fuel, alpha, max_thrust], r=penalty_function, maxiter=10)

print([m.value, max_fuel.value, alpha.value, max_thrust.value])
print(prob.solve())
