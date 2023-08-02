#pragma once

#include <unordered_map>

#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/scs_solver.h"

using SolverIdToSolverUnorderdMap =
    std::unordered_map<drake::solvers::SolverId,
                       std::unique_ptr<drake::solvers::SolverInterface>>;

/*
 * This class owns every solver.
 */
class SolverSelector {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SolverSelector)
  static std::unique_ptr<SolverSelector> MakeSolverSelector();
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

  [[nodiscard]] const drake::solvers::SolverInterface& PickBestSocpSolver(
      bool needs_dual) const;
  [[nodiscard]] const drake::solvers::SolverInterface& PickBestQpSolver() const;

  [[nodiscard]] const drake::solvers::SolverInterface& get_solver(
      const drake::solvers::SolverId& solver_id) const {
    return *solver_id_to_solver_map_.at(solver_id);
  }

 private:
  explicit SolverSelector(
      SolverIdToSolverUnorderdMap&& solver_id_to_solver_map);
  const SolverIdToSolverUnorderdMap solver_id_to_solver_map_;
  const bool has_gurobi_{false};
  const bool has_mosek_{false};
};
