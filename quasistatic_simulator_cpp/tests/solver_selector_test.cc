#include "diffcp/solver_selector.h"

#include <gtest/gtest.h>

class SolverSelectorTest : public testing::Test {
 protected:
  std::unique_ptr<SolverSelector> solver_selector_ =
      SolverSelector::MakeSolverSelector();
};

TEST_F(SolverSelectorTest, PickBestSocp) {
  EXPECT_EQ(SolverSelector::PickBestSocpSolver(false, false, false),
            drake::solvers::ScsSolver::id());
  EXPECT_EQ(SolverSelector::PickBestSocpSolver(false, false, true),
            drake::solvers::ScsSolver::id());
  EXPECT_EQ(SolverSelector::PickBestSocpSolver(false, true, false),
            drake::solvers::MosekSolver::id());
  EXPECT_EQ(SolverSelector::PickBestSocpSolver(false, true, true),
            drake::solvers::MosekSolver::id());
  EXPECT_EQ(SolverSelector::PickBestSocpSolver(true, false, false),
            drake::solvers::GurobiSolver::id());
  EXPECT_EQ(SolverSelector::PickBestSocpSolver(true, false, true),
            drake::solvers::ScsSolver::id());
  EXPECT_EQ(SolverSelector::PickBestSocpSolver(true, true, false),
            drake::solvers::GurobiSolver::id());
  EXPECT_EQ(SolverSelector::PickBestSocpSolver(true, true, true),
            drake::solvers::MosekSolver::id());
}

TEST_F(SolverSelectorTest, PickBestQp) {
  EXPECT_EQ(SolverSelector::PickBestQpSolver(false, false),
            drake::solvers::OsqpSolver::id());
  EXPECT_EQ(SolverSelector::PickBestQpSolver(false, true),
            drake::solvers::MosekSolver::id());
  EXPECT_EQ(SolverSelector::PickBestQpSolver(true, false),
            drake::solvers::GurobiSolver::id());
  EXPECT_EQ(SolverSelector::PickBestQpSolver(true, true),
            drake::solvers::GurobiSolver::id());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
