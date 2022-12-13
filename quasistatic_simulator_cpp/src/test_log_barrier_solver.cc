#include <gtest/gtest.h>

#include "drake/math/jacobian.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"

#include "log_barrier_solver.h"
#include "test_utilities.h"

using drake::AutoDiffXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/*
 * Test barrier solvers on the ball-box grazing problem.
 */
class TestLogBarrierSolvers : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    use_free_solver_ = GetParam();

    Q_.resize(n_v_, n_v_);
    Q_.setIdentity();

    // A commanded position that could lead to problems.
    // TODO: add more tests.
    tau_h_.resize(n_v_);
    tau_h_ << -0.41669826, -0.46413128, 0;

    phi_pyramid_.resize(2);
    phi_pyramid_.setZero();
    phi_icecream_.resize(1);
    phi_icecream_.setZero();

    MatrixXd Jn(1, n_v_);
    Jn << 0, 1, 0;
    MatrixXd Jt(1, n_v_);
    Jt << -1, 0, 1;

    J_pyramid_.resize(2, n_v_);
    J_pyramid_.row(0) = Jn + mu_ * Jt;
    J_pyramid_.row(1) = Jn - mu_ * Jt;

    J_icecream_.resize(3, n_v_);
    J_icecream_.row(0) = Jn / mu_;
    J_icecream_.row(1) = Jt;
    J_icecream_.row(2).setZero();
  }

  bool use_free_solver_{false};
  const int n_v_{3};
  const double mu_{1}, kappa_{100}, h_{0.1};
  MatrixXd Q_, J_pyramid_, J_icecream_;
  VectorXd tau_h_, phi_pyramid_, phi_icecream_;
};

TEST_P(TestLogBarrierSolvers, TestSocpGradientAndHessian) {
  // Test with Hessian and Gradients computed using drake.
  // Wraps cost function in a lambda function that can be consumed by
  //  drake::math::hessian(...).
  auto f = [&](const auto &x) {
    using Scalar = typename std::remove_reference_t<decltype(x)>::Scalar;
    drake::Vector1<Scalar> output;
    drake::VectorX<Scalar> phi_h_mu(phi_icecream_.size());
    for (int i = 0; i < phi_icecream_.size(); i++) {
      phi_h_mu[i] = phi_icecream_[i] / h_ / mu_;
    }
    output[0] = SocpLogBarrierSolver::DoCalcF<Scalar>(
        Q_.template cast<Scalar>(), -tau_h_.template cast<Scalar>(),
        -J_icecream_.template cast<Scalar>(), phi_h_mu, kappa_, x);
    return output;
  };

  VectorXd v(3);
  v << 0.5, 2, 1;

  auto f_value = drake::math::hessian(f, v)[0];
  VectorXd Df_drake(n_v_);
  MatrixXd H_drake(n_v_, n_v_);
  for (int i = 0; i < n_v_; i++) {
    Df_drake[i] = f_value.derivatives()[i].value();
    H_drake.row(i) = f_value.derivatives()[i].derivatives();
  }

  auto solver_log_socp = SocpLogBarrierSolver();
  Eigen::VectorXd Df(n_v_);
  Eigen::MatrixXd H(n_v_, n_v_);
  solver_log_socp.CalcGradientAndHessian(
      Q_, -tau_h_, -J_icecream_, phi_icecream_ / mu_ / h_, v, kappa_, &Df, &H);

  EXPECT_LT((Df_drake - Df).norm(), 1e-8);
  EXPECT_LT((H_drake - H).norm(), 1e-8);
}

TEST_P(TestLogBarrierSolvers, TestSolve) {
  auto solver_pyramid = QpLogBarrierSolver();
  auto solver_icecream = SocpLogBarrierSolver();
  // TODO: also compare with MOSEK.
  VectorXd v_star_pyramid, v_star_icecream;
  solver_pyramid.Solve(Q_, -tau_h_, -J_pyramid_, phi_pyramid_ / h_, kappa_,
                       use_free_solver_, &v_star_pyramid);
  solver_icecream.Solve(Q_, -tau_h_, -J_icecream_, phi_icecream_ / mu_ / h_,
                        kappa_, use_free_solver_, &v_star_icecream);
  EXPECT_LT((v_star_pyramid - v_star_icecream).norm(), 1e-5);
}

TEST_P(TestLogBarrierSolvers, TestMultipleStepNewton) {
  auto solver_pyramid = QpLogBarrierSolver();
  auto solver_icecream = SocpLogBarrierSolver();

  VectorXd v_star_pyramid(n_v_), v_star_icecream(n_v_);
  solver_pyramid.SolvePhaseOne(-J_pyramid_, phi_pyramid_ / h_, use_free_solver_,
                               &v_star_pyramid);
  solver_icecream.SolvePhaseOne(-J_icecream_, phi_icecream_ / mu_ / h_,
                                use_free_solver_, &v_star_icecream);

  solver_pyramid.SolveMultipleNewtonSteps(
      Q_, -tau_h_, -J_pyramid_, phi_pyramid_ / h_, kappa_, &v_star_pyramid);
  solver_icecream.SolveMultipleNewtonSteps(Q_, -tau_h_, -J_icecream_,
                                           phi_icecream_ / mu_ / h_, kappa_,
                                           &v_star_icecream);
  EXPECT_LT((v_star_pyramid - v_star_icecream).norm(), 1e-5);
}

TEST_P(TestLogBarrierSolvers, TestGradientDescent) {
  auto solver_pyramid = QpLogBarrierSolver();
  auto solver_icecream = SocpLogBarrierSolver();

  VectorXd v_star_pyramid(n_v_), v_star_icecream(n_v_);
  solver_pyramid.SolvePhaseOne(-J_pyramid_, phi_pyramid_ / h_, use_free_solver_,
                               &v_star_pyramid);
  solver_icecream.SolvePhaseOne(-J_icecream_, phi_icecream_ / mu_ / h_,
                                use_free_solver_, &v_star_icecream);

  solver_pyramid.SolveGradientDescent(
      Q_, -tau_h_, -J_pyramid_, phi_pyramid_ / h_, kappa_, &v_star_pyramid);
  solver_icecream.SolveGradientDescent(Q_, -tau_h_, -J_icecream_,
                                       phi_icecream_ / mu_ / h_, kappa_,
                                       &v_star_icecream);
  EXPECT_LT((v_star_pyramid - v_star_icecream).norm(), 1e-5);
}

INSTANTIATE_TEST_SUITE_P(
    LogBarrierSolver, TestLogBarrierSolvers,
    testing::ValuesIn(qsim::test::get_use_free_solvers_values()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
