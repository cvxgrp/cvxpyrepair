import cvxpy as cp
import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict

from cvxpyrepair.diffcprepair import derivative


def repair(prob, params, r=None, verbose=True, maxiter=10, maxiter_pgm=25,
           lam=1, lam_factor=2, lr=.1):
    """
    Repairs prob by altering params.
    Minimizes r(params) subject to the constraint that prob is solvable.

    Args:
        - prob: cvxpy Problem object
        - params: list of cvxpy Parameters involved in prob
        - r (optional): callable that takes a list of cvxpy Variables as input
            (of the same dimension as params), and returns a cvxpy expression
            representing the performance metric (default=None).
        - verbose (optional): whether or not to print diagnostic informtion (default=True).
        - maxiter (optional): Maximum number of outer iterations (default=10).
        - maxiter_pgm (optional): Maximum number of inner iterations (default=25).
        - lam (optional): Starting value for 1/lambda, multiplied by lam_factor
            each iteration (default=1).
        - lam_factor (optional): Factor to multiply lambda by each iteration (default=2). 
        - lr (optional): initial step size for proximal gradient method (default=.1).
    """
    assert set(prob.parameters()) == set(params)
    assert hasattr(prob, "get_problem_data")

    # compile problem
    data, _, _ = prob.get_problem_data(solver=cp.SCS)
    compiler = data[cp.settings.PARAM_PROB]
    cone_dict = dims_to_solver_dict(data["dims"])
    param_ids = [p.id for p in params]
    warm_start = None

    for k in range(maxiter):
        # minimize (1/lam) * r(A, b, c) + t(A, b, c)
        objective = float("inf")

        for k_pgm in range(maxiter_pgm):
            # compute derivative
            c, _, neg_A, b = compiler.apply_parameters(
                dict(zip(param_ids, [p.value for p in params])))
            A = -neg_A
            dA, db, dc, t, _, _, _, warm_start = derivative(
                A, b, c, cone_dict, warm_start=warm_start, acceleration_lookback=0, eps=1e-8)
            del_param_dict = compiler.apply_param_jac(dc, -dA, db)
            param_derivative = [del_param_dict[i] for i in param_ids]

            # compute objective
            objective = t
            new_objective = float("inf")
            if r is not None:
                variable_params = [cp.Variable(p.shape) for p in params]
                for vp, p in zip(variable_params, params):
                    vp.value = p.value
                obj, cons = r(variable_params)
                objective += (1 / lam) * obj.value

            # update step size until objective decreases
            old_params = [p.value.copy() for p in params]
            while True:
                lr = np.clip(lr, 1e-6, 1e6)
                new_params = []
                for i in range(len(param_ids)):
                    new_params += [old_params[i] -
                                   lr * param_derivative[i]]

                print(old_params[0], param_derivative[0])
                if r is not None:
                    variable_params = [cp.Variable(
                        p.shape) for p in params]
                    obj, cons = r(variable_params)
                    obj = lr * obj / lam
                    obj += cp.sum([.5 * cp.sum_squares(p - v)
                                   for (p, v) in zip(new_params, variable_params)])
                    prob = cp.Problem(cp.Minimize(obj), cons)
                    try:
                        prob.solve(solver=cp.MOSEK)
                    except:
                        prob.solve(solver=cp.SCS, acceleration_lookback=0)
                    for i in range(len(param_ids)):
                        params[i].value = variable_params[i].value
                else:
                    for i in range(len(param_ids)):
                        params[i].value = new_params[i]

                # compute objective
                c, _, neg_A, b = compiler.apply_parameters(
                    dict(zip(param_ids, [p.value for p in params])))
                A = -neg_A
                _, _, _, t_new, _, _, _, warm_start = derivative(
                    A, b, c, cone_dict, warm_start=warm_start, acceleration_lookback=0, eps=1e-8)
                new_objective = t_new
                if r is not None:
                    variable_params = [cp.Variable(p.shape) for p in params]
                    for vp, p in zip(variable_params, params):
                        vp.value = p.value
                    obj, cons = r(variable_params)
                    new_objective += obj.value / lam

                if new_objective < objective:
                    lr *= 1.2
                    break
                elif lr > 1e-6:
                    lr /= 2.0
                else:
                    break

            if lr <= 1e-6:
                break

        # update lam
        lam *= lam_factor
        if verbose:
            print(f'Updating lambda to {lam}')

        r_val = 0.0
        if r is not None:
            variable_params = [cp.Variable(p.shape) for p in params]
            for vp, p in zip(variable_params, params):
                vp.value = p.value
            obj, cons = r(variable_params)
            r_val += obj.value
        if verbose:
            print("Proximal gradient method completed in %d/%d iterations" %
                  (k_pgm + 1, maxiter_pgm))
            print("Iteration: %d, r: %3.3f, t: %3.3f" % (k, r_val, t))
