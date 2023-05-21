#pragma once
#include <Eigen/Dense>
#include <vector>

/*
QP:
min. 1 / 2 * z.dot(Q).dot(z) + b.dot(z)
s.t. G.dot(z) <= e
 */

class QpDerivativesBase {
 public:
  explicit QpDerivativesBase(double tol) : tol_(tol) {}
  [[nodiscard]] const Eigen::MatrixXd& get_DzDe() const { return DzDe_; }
  [[nodiscard]] const Eigen::MatrixXd& get_DzDb() const { return DzDb_; }
  static void CheckSolutionError(double error, double tol, int n);
  static Eigen::MatrixXd CalcInverseAndCheck(
      const Eigen::Ref<const Eigen::MatrixXd>& A, double tol);

 protected:
  const double tol_;
  Eigen::MatrixXd DzDe_;
  Eigen::MatrixXd DzDb_;
};

class QpDerivatives : public QpDerivativesBase {
 public:
  explicit QpDerivatives(double tol) : QpDerivativesBase(tol) {}
  void UpdateProblem(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                     const Eigen::Ref<const Eigen::VectorXd>& b,
                     const Eigen::Ref<const Eigen::MatrixXd>& G,
                     const Eigen::Ref<const Eigen::MatrixXd>& e,
                     const Eigen::Ref<const Eigen::VectorXd>& z_star,
                     const Eigen::Ref<const Eigen::VectorXd>& lambda_star);
};

class QpDerivativesActive : public QpDerivativesBase {
 public:
  explicit QpDerivativesActive(double tol) : QpDerivativesBase(tol) {}

  /*
   * For computing the derivatives of contact dynamics formulated as a QP,
   * calc_G_grad is false when only the derivatives w.r.t. q_a_cmd
   * (actuation) is needed. In contrast, it is true when the derivatives
   * w.r.t. q (state) is also needed.
   */
  void UpdateProblem(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                     const Eigen::Ref<const Eigen::VectorXd>& b,
                     const Eigen::Ref<const Eigen::MatrixXd>& G,
                     const Eigen::Ref<const Eigen::MatrixXd>& e,
                     const Eigen::Ref<const Eigen::VectorXd>& z_star,
                     const Eigen::Ref<const Eigen::VectorXd>& lambda_star,
                     double lambda_threshold, bool calc_G_grad);
  [[nodiscard]] std::pair<const Eigen::MatrixXd&, const std::vector<int>&>
  get_DzDvecG_active() const {
    return {DzDvecG_active_, lambda_star_active_indices_};
  }

 private:
  Eigen::MatrixXd DzDvecG_active_;
  std::vector<int> lambda_star_active_indices_;
};
