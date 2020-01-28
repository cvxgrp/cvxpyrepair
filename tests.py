import unittest

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime, check_grad
from cvxpyrepair import repair, derivative

class TestCvxpyRepair(unittest.TestCase):
    def test_smoke_test(self):
        x = cp.Variable(1)
        a = cp.Parameter(1)
        prob = cp.Problem(cp.Minimize(x), [x >= 1 + a, x <= 1 - a])
        a.value = np.array([1.0])

        repair(prob, [a], verbose=False)
        self.assertLessEqual(a.value[0], 0.0)

class TestDiffcpRepair(unittest.TestCase):
    def test_derivative(self):
        np.random.seed(0)

        cone_dict = {
            "f": 1,
            "l": 1,
            "q": [1],
            "s": [2],
            "ep": 1,
            "ed": 1
        }

        m = 1 + 1 + 3 + 1 + 2*3
        n = 3

        A = sparse.csc_matrix(np.random.randn(m, n))
        b = np.random.randn(m)
        c = np.random.randn(n)


        def func(x):
            A_data, b_data, c_data = np.split(x, [A.nnz, A.nnz + m])
            A.data = A_data
            b = b_data
            c = c_data
            dA, db, dc, objective, x, y, s = derivative(A, b, c, cone_dict, eps=1e-12)
            return objective

        def grad(x):
            A_data, b_data, c_data = np.split(x, [A.nnz, A.nnz + m])
            A.data = A_data
            b = b_data
            c = c_data
            dA, db, dc, objective, x, y, s = derivative(A, b, c, cone_dict, eps=1e-12)
            return np.concatenate([dA.data, db, dc])

        self.assertLessEqual(check_grad(func, grad, np.concatenate([A.data, b, c]), epsilon=1e-6) / (m*n+m+n), 1e-3)

if __name__ == '__main__':
    unittest.main()