#include <vector>
#include "test_utilities.h"

#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"

std::vector<bool> qsim::test::get_use_free_solvers_values() {
  if (drake::solvers::GurobiSolver::is_available() and
      drake::solvers::MosekSolver::is_available()) {
    // Test with both free and commercial solvers.
    return {true, false};
  }
  // Test only with free solvers.
  return {true};
}