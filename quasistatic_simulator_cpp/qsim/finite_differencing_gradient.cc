#include "qsim/finite_differencing_gradient.h"

#include "drake/solvers/mathematical_program.h"

Eigen::MatrixXd FiniteDiffGradientCalculator::CalcB(
    const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
    const Eigen::Ref<const Eigen::VectorXd>& u_nominal, const double du,
    QuasistaticSimParameters sim_params) {
  const auto n_x = x_nominal.size();
  const auto n_u = u_nominal.size();
  Eigen::MatrixXd DfDu(n_x, n_u);

  sim_params.gradient_mode = GradientMode::kNone;

  for (int i_u = 0; i_u < n_u; i_u++) {
    Eigen::VectorXd u_plus(u_nominal), u_minus(u_nominal);
    u_plus[i_u] += du;
    u_minus[i_u] -= du;

    auto x_next_plus = q_sim_->CalcDynamics(x_nominal, u_plus, sim_params);
    auto x_next_minus = q_sim_->CalcDynamics(x_nominal, u_minus, sim_params);
    DfDu.col(i_u) = (x_next_plus - x_next_minus) / 2 / du;
  }

  return DfDu;
}

Eigen::MatrixXd FiniteDiffGradientCalculator::CalcA(
    const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
    const Eigen::Ref<const Eigen::VectorXd>& u_nominal, const double dx,
    QuasistaticSimParameters sim_params) {
  const auto n_x = x_nominal.size();
  Eigen::MatrixXd DfDx(n_x, n_x);

  sim_params.gradient_mode = GradientMode::kNone;

  for (int i = 0; i < n_x; i++) {
    Eigen::VectorXd x_plus(x_nominal), x_minus(x_nominal);
    x_plus[i] += dx;
    x_minus[i] -= dx;

    auto x_next_plus = q_sim_->CalcDynamics(x_plus, u_nominal, sim_params);
    auto x_next_minus = q_sim_->CalcDynamics(x_minus, u_nominal, sim_params);
    DfDx.col(i) = (x_next_plus - x_next_minus) / 2 / dx;
  }

  return DfDx;
}
