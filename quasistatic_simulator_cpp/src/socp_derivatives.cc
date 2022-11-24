#include <iostream>

#include <unsupported/Eigen/KroneckerProduct>

#include "socp_derivatives.h"

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::seq;
using Eigen::seqN;
using Eigen::VectorXd;
using std::cout;
using std::endl;

MatrixXd CalcC(const Eigen::Ref<const Eigen::VectorXd> &v) {
  const auto n = v.size();
  MatrixXd V(n, n);
  V.setZero();
  V.row(0) = v;
  V(seq(1, Eigen::last), 0) = v.tail(n - 1);
  V(seq(1, Eigen::last), seq(1, Eigen::last)).diagonal().setConstant(v[0]);
  return V;
}

/*
 * A12: (n_z, n_c_active * m)
 * C_lambda_list: n_c_active (m, m) matrices.
 */
MatrixXd CalcA12CLambda(const Eigen::Ref<const Eigen::MatrixXd> &A12,
                        const std::vector<MatrixXd> &C_lambda_list) {
  MatrixXd A12C_lambda(A12);
  const auto n_c = C_lambda_list.size();
  const auto m = C_lambda_list[0].rows();
  for (int i = 0; i < n_c; i++) {
    auto i_start = i * m;
    A12C_lambda(Eigen::all, seqN(i_start, m)) =
        A12(Eigen::all, seqN(i_start, m)) * C_lambda_list[i];
  }

  return A12C_lambda;
}

void SocpDerivatives::UpdateProblem(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &b,
    const std::vector<Eigen::MatrixXd> &G_list,
    const std::vector<Eigen::VectorXd> &e_list,
    const Eigen::Ref<const Eigen::VectorXd> &z_star,
    const std::vector<Eigen::VectorXd> &lambda_star_list,
    double lambda_threshold, bool calc_G_grad) {
  const auto n_z = z_star.size();
  const auto n_c = lambda_star_list.size();
  const auto m = e_list[0].size(); // For contact problems, m == 3.
  lambda_star_active_indices_.clear();

  // Find active constraints with large lagrange multipliers.
  for (int i = 0; i < n_c; i++) {
    const auto &lambda_star_i = lambda_star_list[i];
    if (lambda_star_i.norm() > lambda_threshold) {
      lambda_star_active_indices_.push_back(i);
    }
  }
  const auto n_c_active = lambda_star_active_indices_.size();

  // Form A_inv and find A using pseudo-inverse.
  // Length of all active Lagrange multipliers combined.
  const auto n_la = n_c_active * m;
  const auto n_A = n_z + n_la;
  MatrixXd A_inv(n_A, n_A);
  A_inv.setZero();
  A_inv.topLeftCorner(n_z, n_z) = Q;

  std::vector<MatrixXd> C_lambda_list;
  VectorXd lambda_star_active(n_la);
  for (int i = 0; i < n_c_active; i++) {
    const auto idx = lambda_star_active_indices_[i];
    const MatrixXd &G_i = G_list[idx];
    const VectorXd &e_i = e_list[idx];
    const VectorXd &lambda_i = lambda_star_list[idx];

    VectorXd w_i = -G_i * z_star + e_i;
    MatrixXd C_lambda_i = CalcC(lambda_i);

    const auto k = n_z + m * i;
    A_inv.block(0, k, n_z, m) = G_i.transpose();
    A_inv.block(k, 0, m, n_z) = -C_lambda_i * G_i;
    A_inv.block(k, k, m, m) = CalcC(w_i);

    C_lambda_list.emplace_back(std::move(C_lambda_i));
    lambda_star_active(seqN(m * i, m)) = lambda_i;
  }

  const MatrixXd A = QpDerivatives::CalcInverseAndCheck(A_inv, tol_);
  //  cout << "A_inv\n" << A_inv << endl;
  //  cout << "A\n" << A << endl;

  const MatrixXd &A_11 = A.topLeftCorner(n_z, n_z);
  DzDb_ = -A_11;

  DzDe_ = MatrixXd::Zero(n_z, n_c * m);
  for (int i = 0; i < n_c_active; i++) {
    const auto idx = lambda_star_active_indices_[i];
    DzDe_(Eigen::all, seqN(idx * m, m)) =
        -A.block(0, n_z + i * m, n_z, m) * C_lambda_list[i];
    //    cout << "i: " << i << " idx: "<< idx << endl << C_lambda_list[i] <<
    //    endl;
  }

  // DzDvecG_active
  if (not calc_G_grad) {
    return;
  }

  if (lambda_star_active_indices_.empty()) {
    DzDvecG_active_.resize(n_z, 0);
    return;
  }

  const MatrixXd &A_12 = A.topRightCorner(n_z, n_la);
  MatrixXd A12CLambda = CalcA12CLambda(A_12, C_lambda_list);
  DzDvecG_active_ =
      -Eigen::kroneckerProduct(A_11, lambda_star_active.transpose());
  DzDvecG_active_ += Eigen::kroneckerProduct(z_star.transpose(), A12CLambda);
}
