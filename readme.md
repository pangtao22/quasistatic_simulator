# Quasistatic Simulator

Implementation of the paper ["A Convex Quasistatic Time-stepping Scheme for Rigid Multibody Systems with Contact and Friction"](http://groups.csail.mit.edu/robotics-center/public_papers/Pang20b.pdf), published at ICRA2021. Still a work in progress, but many (6) unit tests are passing using the latest version of [Drake](https://drake.mit.edu). 

Some interactive animations generated using the code in this repo can be found in [this slide deck](https://slides.com/pang/deck-28a801).

## Dependencies
- Drake **built with Gurobi** (replacing the QP solver with OSQP should work too).
- cvxpy.
- [iiwa_controller](https://github.com/pangtao22/iiwa_controller) repo (make sure it's on `PYTHONPATH`).
- Possibly others (if import fails...)


## Running the tests
In the root of the repo, run 
```bash
pytest .
```

