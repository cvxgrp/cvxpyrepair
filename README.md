# cvxpyrepair

cvxpyrepair is a Python package that implements the algorithms described in our paper [Automatic repair of convex optimization problems](https://web.stanford.edu/~sbarratt/auto_repair_cvx.pdf).

## Installation

To install, run:
```bash
pip install .
```

## Running the Examples

Navigate to the `examples` folder.

To run the spacecraft landing example:
```
python spacecraft_landing.py
```

To run the arbitrage example:
```
python arbitrage.py
```

To run the boolean LP example:
```
python boolean_lp.py
```

## API

The package has a single API, the `repair` function, which takes in a CVXPY
problem, a list of parameters, and a regularization function, and
tries to repair the problem by changing the parameters and keeping the
regularization function small.
The `repair` function has the signature:
```
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
```

## License

This repository carries an Apache 2.0 license.

## Citing

If you use `cvxpyrepair` in your research, please consider citing our accompanying paper:
```
@article{barratt2020automatic,
  author={Barratt, S. and Angeris, G. and Boyd, S.},
  title={Automatic Repair of Convex Optimization Problems},
  year={2020},
  journal={Manuscript}
}
```