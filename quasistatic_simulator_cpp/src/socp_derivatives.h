#pragma once

#include "qp_derivatives.h"

/*
 * Consider the SOCP
 * min_{z}. 0.5 * z.T * Q * z + b.T * z
 *  s.t. -G_i * z + e_i \in Q^m,
 *  where z is a (n_z)-vector; Q^m is the m-dimensional second-order cone; G_i
 *  is a (m, n_z) matrix; e_i is an m-vector.
 */
class SocpDerivatives {
public:
  explicit SocpDerivatives(double tol) : tol_(tol){};
  void UpdateProblem(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                     const Eigen::Ref<const Eigen::VectorXd> &b,
                     const std::vector<Eigen::MatrixXd> &G_list,
                     const std::vector<Eigen::VectorXd> &e_list,
                     const Eigen::Ref<const Eigen::VectorXd> &z_star,
                     const std::vector<Eigen::VectorXd> &lambda_star_list,
                     double lambda_threshold, bool calc_G_grad);
  [[nodiscard]] const Eigen::MatrixXd &get_DzDe() const { return DzDe_; };
  [[nodiscard]] const Eigen::MatrixXd &get_DzDb() const { return DzDb_; };
  [[nodiscard]] std::pair<const Eigen::MatrixXd &, const std::vector<int> &>
  get_DzDvecG_active() const {
    return {DzDvecG_active_, lambda_star_active_indices_};
  }
private:
  const double tol_;
  Eigen::MatrixXd DzDe_;
  Eigen::MatrixXd DzDb_;
  Eigen::MatrixXd DzDvecG_active_;
  std::vector<int> lambda_star_active_indices_;
};
