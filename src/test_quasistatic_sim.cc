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

class TestQuasistaticSim : public ::testing::Test {
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

    const auto n_q = q_sim_->get_plant().num_positions();
    const auto n_a = q_sim_->num_actuated_dofs();
    q0_.resize(n_q);
    u0_.resize(n_a);
    q0_ << 3.50150400e-02, 7.52765650e-01, 7.41462320e-01, 8.32610020e-01,
        6.32562690e-01, 1.02378254e+00, 6.40895550e-01, 8.24447820e-01,
        -1.43872500e-01, 7.46968120e-01, 6.19088270e-01, 7.00642790e-01,
        -6.92254100e-02, 7.85331420e-01, 8.29428630e-01, 9.04154360e-01,
        0.98334744, 0.0342708, 0.10602051, 0.14357218, //  <- quaternion.
        -9.00000000e-02, 1.00000000e-03, 8.00000000e-02;

    u0_ << 0.03501504, 0.75276565, 0.74146232, 0.83261002, 0.63256269,
        1.02378254, 0.64089555, 0.82444782, -0.1438725, 0.74696812, 0.61908827,
        0.70064279, -0.06922541, 0.78533142, 0.82942863, 0.90415436;
  }

  QuasistaticSimParameters params_;
  std::unique_ptr<QuasistaticSimulator> q_sim_;
  std::unique_ptr<BatchQuasistaticSimulator> q_sim_b_;
  VectorXd q0_, u0_;
};

TEST_F(TestQuasistaticSim, TestDfDu) {
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
  const auto B_numerical = fd.CalcB(q0_, u0_, 5e-4, params_);
  auto A_numerical = fd.CalcA(q0_, u0_, 5e-4, params_);
  SetSmallNumbersToZero(A_numerical, 1e-10);

  const auto &[B_zero, c_zero] = q_sim_b_->CalcBcLstsq(
      q0_, u0_, params_, Eigen::VectorXd::Constant(u0_.size(), 0.1), 1000);

  cout << "Norm diff A " << (A_analytic_socp - A_analytic_qp).norm();
  cout << " A_numerical_norm " << A_numerical.norm();
  cout << " A_analytic_socp_norm " << A_analytic_socp.norm();
  cout << " A_analytic_qp_norm " << A_analytic_qp.norm() << endl;
  cout << "A_analytic_socp\n" << A_analytic_socp << endl;
  cout << "A_analytic_qp\n" << A_analytic_qp << endl;
  cout << "A_numerical\n" << A_numerical << endl;

  cout << "B_analytic_socp\n" << B_analytic_socp << endl;
  cout << "B_zero\n" << B_zero << endl;
  cout << "Norm diff B " << (B_analytic_socp - B_zero).norm() << endl;


}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
