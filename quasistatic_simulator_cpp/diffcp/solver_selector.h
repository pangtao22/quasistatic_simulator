#pragma once

#include "drake/solvers/solve.h"

class SolverSelector {
 public:
  /*
   *  has_gurobi | has_mosek  | needs_dual |  Best Solver
   *      0      |      0     |     0      |      Scs
   *      0      |      0     |     1      |      Scs
   *      0      |      1     |     0      |      Mosek
   *      0      |      1     |     1      |      Mosek
   *      1      |      0     |     0      |      Gurobi
   *      1      |      0     |     1      |      Scs
   *      1      |      1     |     0      |      Gurobi
   *      1      |      1     |     1      |      Mosek
   */
  static drake::solvers::SolverId PickBestSocpSolver(bool has_gurobi,
                                                     bool has_mosek,
                                                     bool needs_dual);
  /*
   *  has_gurobi | has_mosek  |   Best Solver
   *      0      |      0     |       Osqp
   *      0      |      1     |       Mosek
   *      1      |      0     |       Gurobi
   *      1      |      1     |       Gurobi
   */
  static drake::solvers::SolverId PickBestQpSolver(bool has_gurobi,
                                                   bool has_mosek);
};
