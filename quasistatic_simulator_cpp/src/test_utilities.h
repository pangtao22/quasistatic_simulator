#pragma once

#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"

namespace qsim::test {
std::vector<bool> get_use_free_solvers_values();
}
