#include "diffcp/solver_selector.h"

#include <vector>

#include "drake/solvers/choose_best_solver.h"

using drake::solvers::GurobiSolver;
using drake::solvers::MosekSolver;
using drake::solvers::OsqpSolver;
using drake::solvers::ScsSolver;

namespace {
/*
 * Note that solver.available() involves reading the license file, which is
 *  expensive. This function should be called only once during construction.
 */
bool HasSolver(const drake::solvers::SolverInterface& solver) {
  return solver.available() && solver.enabled();
}
}  // namespace

SolverSelector::SolverSelector(
    SolverIdToSolverUnorderdMap&& solver_id_to_solver_map)
    : solver_id_to_solver_map_(std::move(solver_id_to_solver_map)),
      has_gurobi_{HasSolver(*solver_id_to_solver_map_.at(GurobiSolver::id()))},
      has_mosek_{HasSolver(*solver_id_to_solver_map_.at(MosekSolver::id()))} {}

std::unique_ptr<SolverSelector> SolverSelector::MakeSolverSelector() {
  SolverIdToSolverUnorderdMap solver_id_to_solver_map;
  const std::vector<drake::solvers::SolverId> solver_ids{
      ScsSolver::id(), OsqpSolver::id(), MosekSolver::id(), GurobiSolver::id()};
  for (const auto& solver_id : solver_ids) {
    solver_id_to_solver_map[solver_id] = drake::solvers::MakeSolver(solver_id);
  }

  return std::unique_ptr<SolverSelector>(
      new SolverSelector(std::move(solver_id_to_solver_map)));
}

drake::solvers::SolverId SolverSelector::PickBestSocpSolver(bool has_gurobi,
                                                            bool has_mosek,
                                                            bool needs_dual) {
  // Define the truth table outputs for each index
  // clang-format off
  static const std::vector<drake::solvers::SolverId> solver_id_outputs = {
      ScsSolver::id(),
      ScsSolver::id(),
      MosekSolver::id(),
      MosekSolver::id(),
      GurobiSolver::id(),
      ScsSolver::id(),
      GurobiSolver::id(),
      MosekSolver::id()
  };
  // clang-format on
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

const drake::solvers::SolverInterface& SolverSelector::PickBestSocpSolver(
    bool needs_dual) const {
  return *solver_id_to_solver_map_.at(
      PickBestSocpSolver(has_gurobi_, has_mosek_, needs_dual));
}

const drake::solvers::SolverInterface& SolverSelector::PickBestQpSolver()
    const {
  return *solver_id_to_solver_map_.at(
      PickBestQpSolver(has_gurobi_, has_mosek_));
}
