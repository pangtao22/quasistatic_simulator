#include "diffcp/solver_selector.h"

#include <vector>

#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/scs_solver.h"

drake::solvers::SolverId SolverSelector::PickBestSocpSolver(bool has_gurobi,
                                                            bool has_mosek,
                                                            bool needs_dual) {
  // Define the truth table outputs for each index
  static const std::vector<drake::solvers::SolverId> solver_id_outputs = {
      drake::solvers::ScsSolver::id(),    drake::solvers::ScsSolver::id(),
      drake::solvers::MosekSolver::id(),  drake::solvers::MosekSolver::id(),
      drake::solvers::GurobiSolver::id(), drake::solvers::ScsSolver::id(),
      drake::solvers::GurobiSolver::id(), drake::solvers::MosekSolver::id()};

  // Convert the booleans to integers (0 for false, 1 for true)
  const int bit0 = needs_dual ? 1 : 0;
  const int bit1 = has_mosek ? 1 : 0;
  const int bit2 = has_gurobi ? 1 : 0;

  // Calculate the index of the truth table entry using binary representation
  const int index = (bit2 << 2) | (bit1 << 1) | bit0;

  // Return the output corresponding to the calculated index
  return solver_id_outputs[index];
}

drake::solvers::SolverId SolverSelector::PickBestQpSolver(bool has_gurobi,
                                                          bool has_mosek) {
  if (has_gurobi) {
    return drake::solvers::GurobiSolver::id();
  }
  // No gurobi
  if (has_mosek) {
    return drake::solvers::MosekSolver::id();
  }

  // No gurobi or no mosek
  return drake::solvers::OsqpSolver::id();
}
