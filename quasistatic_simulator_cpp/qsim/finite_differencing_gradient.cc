#include "qsim/finite_differencing_gradient.h"

#include "drake/solvers/mathematical_program.h"

FiniteDiffGradientCalculatorBase::FiniteDiffGradientCalculatorBase(int n_x,
                                                                   int n_u)
    : n_x_(n_x), n_u_(n_u) {}

FiniteDiffGradientCalculator::FiniteDiffGradientCalculator(
    QuasistaticSimulator* q_sim)
    : FiniteDiffGradientCalculatorBase(q_sim->get_plant().num_positions(),
                                       q_sim->num_actuated_dofs()),
      q_sim_(q_sim) {}

Eigen::MatrixXd FiniteDiffGradientCalculator::CalcB(
    const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
    const Eigen::Ref<const Eigen::VectorXd>& u_nominal, const double du,
    QuasistaticSimParameters sim_params) {
  Eigen::MatrixXd DfDu(n_x_, n_u_);

  sim_params.gradient_mode = GradientMode::kNone;

  for (int i_u = 0; i_u < n_u_; i_u++) {
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
  Eigen::MatrixXd DfDx(n_x_, n_x_);

  sim_params.gradient_mode = GradientMode::kNone;

  for (int i = 0; i < n_x_; i++) {
    Eigen::VectorXd x_plus(x_nominal), x_minus(x_nominal);
    x_plus[i] += dx;
    x_minus[i] -= dx;

    auto x_next_plus = q_sim_->CalcDynamics(x_plus, u_nominal, sim_params);
    auto x_next_minus = q_sim_->CalcDynamics(x_minus, u_nominal, sim_params);
    DfDx.col(i) = (x_next_plus - x_next_minus) / 2 / dx;
  }

  return DfDx;
}

BatchFiniteDiffGradientCalculator::BatchFiniteDiffGradientCalculator(
    BatchQuasistaticSimulator* q_sim_batch)
    : FiniteDiffGradientCalculatorBase(
          q_sim_batch->get_q_sim().get_plant().num_positions(),
          q_sim_batch->get_q_sim().num_actuated_dofs()),
      q_sim_batch_(q_sim_batch) {
  x_buffer_A_.resize(2 * n_x_, n_x_);
  u_buffer_A_.resize(2 * n_x_, n_u_);
  x_buffer_B_.resize(2 * n_u_, n_x_);
  u_buffer_B_.resize(2 * n_u_, n_u_);
}

Eigen::MatrixXd BatchFiniteDiffGradientCalculator::CalcB(
    const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
    const Eigen::Ref<const Eigen::VectorXd>& u_nominal, const double du,
    QuasistaticSimParameters sim_params) {
  sim_params.gradient_mode = GradientMode::kNone;

  for (int i = 0; i < n_u_; i++) {
    x_buffer_B_.row(i) = x_nominal;
    x_buffer_B_.row(i + n_u_) = x_nominal;

    u_buffer_B_.row(i) = u_nominal;
    u_buffer_B_.row(i + n_u_) = u_nominal;
    u_buffer_B_(i, i) += du;
    u_buffer_B_(i + n_u_, i) -= du;
  }

  auto [x_next_batch, A_batch, B_batch, is_valid_batch] =
      q_sim_batch_->CalcDynamicsParallel(x_buffer_B_, u_buffer_B_, sim_params);

  for (const auto is_valid : is_valid_batch) {
    if (!is_valid) {
      throw std::runtime_error(
          "Dynamics evaluation failed during the computation of B using finite "
          "differencing");
    }
  }

  // DfDu transposed.
  Eigen::MatrixXd DfDu_T =
      (x_next_batch.topRows(n_u_) - x_next_batch.bottomRows(n_u_)) / 2 / du;

  return DfDu_T.transpose();
}

Eigen::MatrixXd BatchFiniteDiffGradientCalculator::CalcA(
    const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
    const Eigen::Ref<const Eigen::VectorXd>& u_nominal, const double dx,
    QuasistaticSimParameters sim_params) {
  sim_params.gradient_mode = GradientMode::kNone;

  for (int i = 0; i < n_x_; i++) {
    x_buffer_A_.row(i) = x_nominal;
    x_buffer_A_.row(i + n_x_) = x_nominal;
    x_buffer_A_(i, i) += dx;
    x_buffer_A_(i + n_x_, i) -= dx;

    u_buffer_A_.row(i) = u_nominal;
    u_buffer_A_.row(i + n_x_) = u_nominal;
  }

  auto [x_next_batch, A_batch, B_batch, is_valid_batch] =
      q_sim_batch_->CalcDynamicsParallel(x_buffer_A_, u_buffer_A_, sim_params);

  for (const auto is_valid : is_valid_batch) {
    if (!is_valid) {
      throw std::runtime_error(
          "Dynamics evaluation failed during the computation of A using finite "
          "differencing");
    }
  }

  Eigen::MatrixXd DfDx_T =
      (x_next_batch.topRows(n_x_) - x_next_batch.bottomRows(n_x_)) / 2 / dx;

  return DfDx_T.transpose();
}
