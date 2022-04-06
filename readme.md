# Quasistatic Simulator

Implementation of the paper ["A Convex Quasistatic Time-stepping Scheme for Rigid Multibody Systems with Contact and Friction"](http://groups.csail.mit.edu/robotics-center/public_papers/Pang20b.pdf), published at ICRA2021. Still a work in progress, but many (>=8) unit tests are passing using the latest version of [Drake](https://drake.mit.edu). 

Some interactive animations generated using the code in this repo can be found in [this slide deck](https://slides.com/pang/deck-28a801).

## Dependencies
- Drake **built with Gurobi** (replacing the QP solver with OSQP should work too).
- cvxpy.
- [iiwa_controller](https://github.com/pangtao22/iiwa_controller) repo (make sure it's on `PYTHONPATH`).
- Possibly others (if import fails...)

## C++ backend
- To use, build [this repo](https://github.com/pangtao22/quasistatic_simulator_cpp.git), and put the built pybind library (which looks like `quasistatic_simulator_py.cpython-39-darwin.so` on a Mac) on `PYTHONPATH`.  

## Running the tests
In the root of the repo, run 
```bash
pytest .
```
Multi-threaded testing with `pytest-xdist `:
```bash
pytest . -n auto
```
