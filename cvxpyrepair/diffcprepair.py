from scipy import sparse
import numpy as np
import diffcp
import cvxpy as cp

ZERO = "f"
FREE = "free"
POS = "l"
SOC = "q"
PSD = "s"
EXP = "ep"
EXP_DUAL = "ed"

# The ordering of CONES matches SCS.
CONES = [ZERO, POS, SOC, PSD, EXP, EXP_DUAL]

DUAL_CONE = {
    ZERO: FREE,
    FREE: ZERO,
    POS: POS,
    SOC: SOC,
    PSD: PSD,
    EXP: EXP_DUAL,
    EXP_DUAL: EXP
}


def dims_from_cone_dict(cone_dict):
    zero_cones = cone_dict.get(ZERO, 0)
    pos_cones = cone_dict.get(POS, 0)
    soc_cones = sum(cone_dict.get(SOC, []))
    psd_cones = sum([vec_psd_dim(c) for c in cone_dict.get(PSD, [])])
    exp_cones = cone_dict.get(EXP, 0) * 3 + cone_dict.get(EXP_DUAL, 0) * 3

    return zero_cones + pos_cones + soc_cones + psd_cones + exp_cones


def dual_cone(cone_dict):
    """Compute the dual cone."""
    dual_cone_dict = {}
    for k, v in cone_dict.items():
        dual_cone_dict[DUAL_CONE[k]] = v
    return dual_cone_dict


def delete_free_cones(cone_dict):
    """Delete free cones."""
    assert ZERO not in cone_dict.keys()
    num_free = cone_dict.get(FREE, 0)
    if num_free > 0:
        del cone_dict[FREE]
    return num_free


def vec_psd_dim(dim):
    """Compute vectorized dimension of dim x dim matrix."""
    return int(dim * (dim + 1) / 2)


def psd_dim(size):
    """Compute non-vectorized dimension of vectorized matrix size."""
    return int(np.sqrt(2 * size))


def derivative(A, b, c, cone_dict, **kwargs):
    """
    Solves the problem

    minimize   \|(Ax+s-b,A^Ty+c,c^Tx+b^Ty)\|_2  (1)
    subject to s in K, y in K^*,

    and computes the derivative of the objective
    with respect to A, b, and c. The objective
    of this problem is 0 if and only if
    (A,b,c,K) form a non-pathological cone program.

    Args:
        - A: m x n matrix 
        - b: m vector
        - c: n vector
        - cone_dict: dict representing K,
            in SCS format

    Returns:
        - dA: m x n matrix with same sparsity pattern as A.
            Derivative of objective with respect to A.
        - db: m vector. Derivative of objective with respect to b.
        - dc: n vector. Derivative of objective with respect to c.
        - objective: Objective value of (1).
        - x: n vector. Solution to (1).
        - y: m vector. Solution to (1).
        - s: m vector. Solution to (1).
    """
    m, n = A.shape

    expected_m = dims_from_cone_dict(cone_dict)
    assert m == expected_m,  "A has %d rows, but should have %d rows." % (
        m, expected_m)
    assert b.size == m
    assert c.size == n

    dual_cone_dict = dual_cone(cone_dict)
    num_free = delete_free_cones(dual_cone_dict)
    Asoc = sparse.bmat([
        [-1.0, None, None, None],
        [None, -A, None, -sparse.eye(m)],
        [None, None, -A.T, None],
        [None, -c[None, :], -b[None, :], None]
    ])
    Ahat = sparse.bmat([
        [sparse.csc_matrix((m, 1)), sparse.csc_matrix(
            (m, n)), None, -sparse.eye(m)],
        [sparse.csc_matrix((m, 1)), sparse.csc_matrix(
            (m, n)), -sparse.eye(m), None]
    ], format='csc')

    # Reformat (Asoc, Ahat) while combining the cones in cone_dict
    # and dual_cone_dict
    pd_cone_dict = {}
    idx = 0
    idx_dual = num_free + m
    mats = []
    inserted_soc_cone = False
    for cone in CONES:
        for dual in [False, True]:
            if cone in [ZERO, POS, EXP, EXP_DUAL]:  # integer cones
                num_cone = (dual_cone_dict if dual else cone_dict).get(cone, 0)
                dim = num_cone
                if cone == EXP or cone == EXP_DUAL:
                    dim *= 3
                if num_cone > 0:
                    pd_cone_dict[cone] = pd_cone_dict.get(cone, 0) + num_cone
                    if not dual:
                        mats.append(Ahat[idx:idx + dim, :])
                        idx += dim
                    else:
                        mats.append(Ahat[idx_dual:idx_dual + dim, :])
                        idx_dual += dim
            else:  # list cones
                if cone == SOC and not inserted_soc_cone:
                    pd_cone_dict[SOC] = [Asoc.shape[0]]
                    mats.append(Asoc)
                    idx_first_soc_cone = idx + idx_dual - m - num_free
                    inserted_soc_cone = True
                cone_list = (
                    dual_cone_dict if dual else cone_dict).get(cone, [])
                dim = sum(cone_list)
                if cone == PSD:
                    dim = sum([vec_psd_dim(c) for c in cone_list])
                if len(cone_list) > 0:
                    pd_cone_dict[cone] = pd_cone_dict.get(cone, []) + cone_list
                    if not dual:
                        mats.append(Ahat[idx:idx + dim, :])
                        idx += dim
                    else:
                        mats.append(Ahat[idx_dual:idx_dual + dim, :])
                        idx_dual += dim

    assert idx == m
    assert idx_dual == 2 * m

    # Prepare (Ahat, bhat, chat)
    Ahat = sparse.vstack(mats, format='csc')
    bhat = np.zeros(Ahat.shape[0])
    bhat[idx_first_soc_cone:idx_first_soc_cone + Asoc.shape[0]] = \
        np.concatenate([np.zeros(1), -b, c, np.zeros(1)])
    chat = np.append(1.0, np.zeros(Ahat.shape[1] - 1))

    # Solve problem and extract optimal value and solution
    x_internal, y_internal, s_internal, _, DT = diffcp.solve_and_derivative(
        Ahat, bhat, chat, pd_cone_dict, **kwargs)
    objective = x_internal[0]
    y = x_internal[1 + n:1 + n + m]
    s = x_internal[1 + n + m:1 + n + 2 * m]
    x = x_internal[1:1 + n]

    # Compute derivatives with respect to Ahat, bhat
    dAhat, dbhat, _ = DT(chat, np.zeros(bhat.size), np.zeros(bhat.size))

    # Extract derivatives with respect to A, b, c from Ahat and bhat
    dAsoc = dAhat[idx_first_soc_cone:idx_first_soc_cone + Asoc.shape[0], :]
    last_row_dAsoc = np.array(dAsoc[-1, :].todense()).ravel()
    dA = -dAsoc[1:1 + m, 1:1 + n] - dAsoc[1 + m:1 + m + n, 1 + n:1 + m + n].T
    db = -last_row_dAsoc[1 + n:1 + n + m]
    db -= dbhat[idx_first_soc_cone + 1:idx_first_soc_cone + 1 + m]
    dc = -last_row_dAsoc[1:1 + n]
    dc += dbhat[idx_first_soc_cone + 1 + m:idx_first_soc_cone + 1 + m + n]

    return dA, db, dc, objective, x, y, s, (x_internal, y_internal, s_internal)
