# Quasistatic Simulator
![ci_badge](https://github.com/pangtao22/quasistatic_simulator/actions/workflows/ci.yml/badge.svg)


![](/media/allegro_hand_ball.gif)  |  ![](/media/allegro_hand_door.gif)

Implementation of the paper ["A Convex Quasistatic Time-stepping Scheme for Rigid Multibody Systems with Contact and Friction"](http://groups.csail.mit.edu/robotics-center/public_papers/Pang20b.pdf), published at ICRA2021. Still a work in progress, but many (>=8) unit tests are passing using the latest version of [Drake](https://drake.mit.edu). 

Some interactive animations generated using the code in this repo can be found in [this slide deck](https://slides.com/pang/deck-28a801).

## Dependencies
- Drake **built with Gurobi and Mosek**. Free solvers (OSQP + SCS) also work, but SCS is a lot slower than Mosek for solving SOCPs.
- In addition to what's in `requirements.txt`, here are two repos that I'll need to include as submodules, but now need to be put manually on `PYTHONPATH`:
  - [iiwa_controller](https://github.com/pangtao22/iiwa_controller) 
  - [manipulation](https://github.com/RussTedrake/manipulation)


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
The python tests right now depend on [a custom branch of drake](https://github.com/pangtao22/drake/tree/my_main).

In the root of the repo, run 
```bash
pytest .
```
Multi-threaded testing with `pytest-xdist `:
```bash
pytest . -n auto
```
