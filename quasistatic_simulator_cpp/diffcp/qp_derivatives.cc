#include "diffcp/qp_derivatives.h"

#include <cmath>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>  // NOLINT

#include "drake/common/drake_assert.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void QpDerivativesBase::CheckSolutionError(const double error, const double tol,
                                           const int n) {
  auto rel_err = error / n;
  bool is_relative_err_small = rel_err < tol;

  if (std::isnan(error)) {
    throw std::runtime_error("Gradient is nan.");
  }
  if (!is_relative_err_small) {
    std::stringstream ss;
    ss << "Relative error " << rel_err << " is greater than " << tol << ".";
    throw std::runtime_error(ss.str());
  }
}

Eigen::MatrixXd QpDerivativesBase::CalcInverseAndCheck(
    const Eigen::Ref<const Eigen::MatrixXd>& A, const double tol) {
  const auto n_A = A.rows();
  DRAKE_ASSERT(n_A == A.cols());
  const auto I = MatrixXd::Identity(n_A, n_A);
  MatrixXd A_inv = Eigen::CompleteOrthogonalDecomposition<MatrixXd>(A).solve(I);
  CheckSolutionError((A_inv * A - I).norm(), tol, n_A);
  return A_inv;
}

void QpDerivatives::UpdateProblem(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::MatrixXd>& e,
    const Eigen::Ref<const Eigen::VectorXd>& z_star,
    const Eigen::Ref<const Eigen::VectorXd>& lambda_star) {
  const size_t n_z = z_star.size();
  const size_t n_l = lambda_star.size();
  MatrixXd A_inv(n_z + n_l, n_z + n_l);
  A_inv.setZero();
  A_inv.topLeftCorner(n_z, n_z) = Q;
  A_inv.topRightCorner(n_z, n_l) = G.transpose();
  A_inv.bottomLeftCorner(n_l, n_z) = lambda_star.asDiagonal() * G;
  A_inv.bottomRightCorner(n_l, n_l).diagonal() = G * (z_star)-e;

  MatrixXd rhs(n_z + n_l, n_z + n_l);
  rhs.setZero();
  rhs.bottomLeftCorner(n_l, n_l).diagonal() = lambda_star;
  rhs.topRightCorner(n_z, n_z).diagonal().setConstant(-1);

  MatrixXd sol = A_inv.colPivHouseholderQr().solve(rhs);
  CheckSolutionError((A_inv * sol - rhs).norm(), tol_, n_z + n_l);

  DzDe_ = sol.topLeftCorner(n_z, n_l);
  DzDb_ = sol.topRightCorner(n_z, n_z);
}

void QpDerivativesActive::UpdateProblem(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& b,
    const Eigen::Ref<const Eigen::MatrixXd>& G,
    const Eigen::Ref<const Eigen::MatrixXd>& e,
    const Eigen::Ref<const Eigen::VectorXd>& z_star,
    const Eigen::Ref<const Eigen::VectorXd>& lambda_star,
    double lambda_threshold, bool calc_G_grad) {
  const auto n_z = z_star.size();
  const auto n_l = lambda_star.size();

  std::vector<double> lambda_star_active_vec;
  lambda_star_active_indices_.clear();

  // Find active constraints with large lagrange multipliers.
  for (int i = 0; i < n_l; i++) {
    double lambda_star_i = lambda_star[i];
    if (lambda_star_i > lambda_threshold) {
      lambda_star_active_vec.push_back(lambda_star_i);
      lambda_star_active_indices_.push_back(i);
    }
  }

  const int n_la = lambda_star_active_vec.size();
  auto lambda_star_active =
      Eigen::Map<VectorXd>(lambda_star_active_vec.data(), n_la);
  MatrixXd B(n_la, n_z);
  for (int i = 0; i < n_la; i++) {
    B.row(i) = G.row(lambda_star_active_indices_[i]);
  }

  // Form A_inv and find A using pseudo-inverse.
  const auto n_A = n_z + n_la;
  MatrixXd A_inv(n_A, n_A);
  A_inv.setZero();
  A_inv.topLeftCorner(n_z, n_z) = Q;
  A_inv.topRightCorner(n_z, n_la) = B.transpose();
  A_inv.bottomLeftCorner(n_la, n_z) = B;
  const MatrixXd A = CalcInverseAndCheck(A_inv, tol_);

  // Compute QP derivatives.
  const MatrixXd& A_11 = A.topLeftCorner(n_z, n_z);
  DzDb_ = -A_11;

  const MatrixXd DzDe_active = A.topRightCorner(n_z, n_la);
  DzDe_ = MatrixXd::Zero(n_z, n_l);
  for (int i = 0; i < n_la; i++) {
    DzDe_.col(lambda_star_active_indices_[i]) = DzDe_active.col(i);
  }

  if (!calc_G_grad) {
    return;
  }

  if (lambda_star_active_indices_.empty()) {
    DzDvecG_active_.resize(n_z, 0);
    return;
  }

  const MatrixXd& A_12 = A.topRightCorner(n_z, n_la);
  DzDvecG_active_ =
      -Eigen::kroneckerProduct(A_11, lambda_star_active.transpose());
  DzDvecG_active_ -= Eigen::kroneckerProduct(z_star.transpose(), A_12);
}
