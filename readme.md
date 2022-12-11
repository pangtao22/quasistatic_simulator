# Quasistatic Simulator
![ci_badge](https://github.com/pangtao22/quasistatic_simulator/actions/workflows/ci.yml/badge.svg)

![](/media/planar_hand.gif) ![](/media/allegro_hand_ball.gif) ![](/media/allegro_hand_door.gif)

This repo provides an implementation of a **differentiable**, **convex** and **quasi-static** dynamics model which is effective for **contact-rich manipulation planning**. The dynamics formulation is described in 
- Section 3 of [Global Planning for Contact-Rich Manipulation via
Local Smoothing of Quasi-dynamic Contact Models](https://arxiv.org/abs/2206.10787), currently under review.
- [A Convex Quasistatic Time-stepping Scheme for Rigid Multibody Systems with Contact and Friction](http://groups.csail.mit.edu/robotics-center/public_papers/Pang20b.pdf), ICRA2021.

Additional interactive animations generated using the code in this repo can be found in [this slide deck](https://slides.com/pang/deck-28a801).

## Dependencies
- Drake **built with Gurobi and Mosek**. Free solvers (OSQP + SCS) also work, but SCS is a lot slower than Mosek for solving SOCPs.

Note that until [this issue](https://github.com/RobotLocomotion/drake-external-examples/issues/216) is resolved, this repo can only be built in debug mode if the official version of Drake is used, which is a lot slower than release mode. A workaround is described in the issue, but requires [a custom branch of drake](https://github.com/pangtao22/drake/tree/my_main).


## Docker
Remember to check out submodules before building the docker images.
```
git submodule update --init --recursive
```

In the root of this repo, to build, run
```
docker build -t qsim -f focal.dockerfile .
```

To run the github "build and test" action locally, run
```
docker run -v $PWD:"/github/workspace" --entrypoint "/github/workspace/scripts/run_tests.sh" qsim
```


## Running python tests
In the root of the repo, run 
```bash
pytest .
```
Multi-threaded testing with `pytest-xdist `:
```bash
pytest . -n auto
```
