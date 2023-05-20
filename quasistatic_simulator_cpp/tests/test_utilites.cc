#include <vector>

#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "tests/test_utilities.h"

std::vector<bool> qsim::test::get_use_free_solvers_values() {
  if (drake::solvers::GurobiSolver::is_available() &&
      drake::solvers::MosekSolver::is_available()) {
    // Test with both free and commercial solvers.
    return {true, false};
  }
  // Test only with free solvers.
  return {true};
}
