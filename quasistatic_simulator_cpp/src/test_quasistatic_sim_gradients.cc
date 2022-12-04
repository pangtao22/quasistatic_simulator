#include <iostream>

#include <gtest/gtest.h>

#include "finite_differencing_gradient.h"
#include "get_model_paths.h"
#include "quasistatic_parser.h"

using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;

void SetSmallNumbersToZero(Eigen::MatrixXd &A, const double threshold = 1e-13) {
  A = (threshold < A.array().abs()).select(A, 0.);
}

class TestQuasistaticSimGradients : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    const string kQModelPath =
        GetQsimModelsPath() / "q_sys" / "allegro_hand_and_sphere.yml";
    auto parser = QuasistaticParser(kQModelPath);

    q_sim_ = parser.MakeSimulator();
    q_sim_b_ = parser.MakeBatchSimulator();

    params_ = parser.get_sim_params();
    params_.h = 0.1;
    params_.gravity = Vector3d(0, 0, -10);
    params_.is_quasi_dynamic = true;
    params_.log_barrier_weight = 100;
    params_.use_free_solvers = GetParam();

    const auto name_to_idx_map = q_sim_->GetModelInstanceNameToIndexMap();
    const string robot_name("allegro_hand_right");
    const string object_name("sphere");
    const auto idx_r = name_to_idx_map.at(robot_name);
    const auto idx_o = name_to_idx_map.at(object_name);
    VectorXd q_u0(7);
    q_u0 << 1, 0, 0, 0, -0.081, 0.001, 0.071;

    VectorXd q_a0(q_sim_->num_actuated_dofs());
    q_a0 << 0.035, 0.753, 0.741, 0.833, -0.144, 0.747, 0.619, 0.701, -0.069,
        0.785, 0.829, 0.904, 0.633, 1.024, 0.641, 0.824;

    ModelInstanceIndexToVecMap q0_dict = {{idx_o, q_u0}, {idx_r, q_a0}};
    q0_ = q_sim_->GetQVecFromDict(q0_dict);
    // +0.05 So that some contacts become active.
    u0_ = q_sim_->GetQaCmdVecFromDict(q0_dict).array() + 0.05;
  }

  QuasistaticSimParameters params_;
  std::unique_ptr<QuasistaticSimulator> q_sim_;
  std::unique_ptr<BatchQuasistaticSimulator> q_sim_b_;
  VectorXd q0_, u0_;
};

TEST_P(TestQuasistaticSimGradients, TestDfDu) {
  params_.gradient_mode = GradientMode::kAB;
  params_.forward_mode = ForwardDynamicsMode::kSocpMp;
  q_sim_->CalcDynamics(q0_, u0_, params_);
  const auto B_analytic_socp = q_sim_->get_Dq_nextDqa_cmd();
  auto A_analytic_socp = q_sim_->get_Dq_nextDq();
  SetSmallNumbersToZero(A_analytic_socp);

  params_.forward_mode = ForwardDynamicsMode::kQpMp;
  q_sim_->CalcDynamics(q0_, u0_, params_);
  const auto B_analytic_qp = q_sim_->get_Dq_nextDqa_cmd();
  auto A_analytic_qp = q_sim_->get_Dq_nextDq();
  SetSmallNumbersToZero(A_analytic_qp);

  // Numerical gradients.
  auto fd = FiniteDiffGradientCalculator(*q_sim_);
  const auto B_numerical = fd.CalcB(q0_, u0_, 1e-3, params_);
  auto A_numerical = fd.CalcA(q0_, u0_, 1e-3, params_);
  SetSmallNumbersToZero(A_numerical, 1e-10);

  // TODO(pang): the bounds are based on running the algorithm on Dec 3, 2022.
  //  It is interesting to investigate why the error in A is so much larger
  //  than in B.
  EXPECT_LT((A_analytic_socp - A_analytic_qp).norm(), 26);
  EXPECT_LT((A_analytic_socp - A_numerical).norm(), 18);
  EXPECT_LT((A_analytic_qp - A_numerical).norm(), 16);
  EXPECT_LT((B_analytic_socp - B_analytic_qp).norm(), 1.5);
  EXPECT_LT((B_analytic_socp - B_numerical).norm(), 1);
  EXPECT_LT((B_analytic_qp - B_numerical).norm(), 1);

  cout << "Norm diff A " << (A_analytic_socp - A_analytic_qp).norm() << endl;
  cout << "Norm diff A2 " << (A_analytic_socp - A_numerical).norm() << endl;
  cout << "Norm diff A3 " << (A_analytic_qp - A_numerical).norm() << endl;
  cout << "Norm diff B " << (B_analytic_socp - B_analytic_qp).norm() << endl;
  cout << "Norm diff B2 " << (B_analytic_socp - B_numerical).norm() << endl;
  cout << "Norm diff B3 " << (B_analytic_qp - B_numerical).norm() << endl;
}

std::vector<bool> get_test_parameters() {
  if (drake::solvers::GurobiSolver::is_available() and
      drake::solvers::MosekSolver::is_available()) {
    // Test with both free and commercial solvers.
    return {true, false};
  }
  // Test only with free solvers.
  return {true};
}

INSTANTIATE_TEST_SUITE_P(QuasistaticSimGradients, TestQuasistaticSimGradients,
                         testing::ValuesIn(get_test_parameters()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
