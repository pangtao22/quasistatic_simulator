#pragma once
#include <Eigen/Dense>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solver_base.h"

class LogBarrierSolver {
public:
  virtual void SolvePhaseOne(const Eigen::Ref<const Eigen::MatrixXd> &G,
                             const Eigen::Ref<const Eigen::VectorXd> &e,
                             drake::EigenPtr<Eigen::VectorXd> v0_ptr) const = 0;

  virtual double CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                       const Eigen::Ref<const Eigen::VectorXd> &b,
                       const Eigen::Ref<const Eigen::MatrixXd> &G,
                       const Eigen::Ref<const Eigen::VectorXd> &e,
                       const double kappa,
                       const Eigen::Ref<const Eigen::VectorXd> &v) const = 0;

  virtual void
  CalcGradientAndHessian(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                         const Eigen::Ref<const Eigen::VectorXd> &b,
                         const Eigen::Ref<const Eigen::MatrixXd> &G,
                         const Eigen::Ref<const Eigen::VectorXd> &e,
                         const Eigen::Ref<const Eigen::VectorXd> &v,
                         double kappa, drake::EigenPtr<Eigen::VectorXd> Df_ptr,
                         drake::EigenPtr<Eigen::MatrixXd> H_ptr) const = 0;

  void GetPhaseOneSolution(const drake::solvers::VectorXDecisionVariable &v,
                           const drake::solvers::DecisionVariable &s,
                           drake::EigenPtr<Eigen::VectorXd> v0_ptr) const;

  void Solve(const Eigen::Ref<const Eigen::MatrixXd> &Q,
             const Eigen::Ref<const Eigen::VectorXd> &b,
             const Eigen::Ref<const Eigen::MatrixXd> &G,
             const Eigen::Ref<const Eigen::VectorXd> &e, double kappa_max,
             Eigen::VectorXd *v_star_ptr) const;

  /*
   * v_star_ptr should come with the starting point. It is then iteratively
   * updated and has the optimal solution when the function returns.
   */
  void SolveOneNewtonStep(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                          const Eigen::Ref<const Eigen::VectorXd> &b,
                          const Eigen::Ref<const Eigen::MatrixXd> &G,
                          const Eigen::Ref<const Eigen::VectorXd> &e,
                          double kappa,
                          drake::EigenPtr<Eigen::VectorXd> v_star_ptr) const;

  /*
   * v_star_ptr should come with the starting point. It is then iteratively
   * updated and has the optimal solution when the function returns.
   */
  void
  SolveMultipleNewtonSteps(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                           const Eigen::Ref<const Eigen::VectorXd> &b,
                           const Eigen::Ref<const Eigen::MatrixXd> &G,
                           const Eigen::Ref<const Eigen::VectorXd> &e,
                           double kappa_max,
                           drake::EigenPtr<Eigen::VectorXd> v_star_ptr) const;

  void SolveGradientDescent(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                            const Eigen::Ref<const Eigen::VectorXd> &b,
                            const Eigen::Ref<const Eigen::MatrixXd> &G,
                            const Eigen::Ref<const Eigen::VectorXd> &e,
                            double kappa,
                            drake::EigenPtr<Eigen::VectorXd> v_star_ptr) const;

  double BackStepLineSearch(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                            const Eigen::Ref<const Eigen::VectorXd> &b,
                            const Eigen::Ref<const Eigen::MatrixXd> &G,
                            const Eigen::Ref<const Eigen::VectorXd> &e,
                            const Eigen::Ref<const Eigen::VectorXd> &v,
                            const Eigen::Ref<const Eigen::VectorXd> &dv,
                            const Eigen::Ref<const Eigen::VectorXd> &Df,
                            const double kappa) const;

  const Eigen::LLT<Eigen::MatrixXd> &get_H_llt() const { return H_llt_; };

protected:
  std::unique_ptr<drake::solvers::SolverBase> solver_;
  mutable drake::solvers::MathematicalProgramResult mp_result_;

  // Hyperparameters for line search.
  static constexpr double alpha_{0.4};
  static constexpr double beta_{0.5};
  static constexpr int line_search_iter_limit_{30};

  // Hyperparameters for Newton's method.
  static constexpr int newton_steps_limit_{50};
  // Considered converge if Newton's decrement / 2 < tol_.
  static constexpr double tol_{1e-6};

  static constexpr int gradient_steps_limit_{500};

private:
  mutable Eigen::LLT<Eigen::MatrixXd> H_llt_;
};

/*
 * Consider the QP
 * min. 0.5 * v.T * Q * v + b.T * v
 *  s.t. G * v - e <= 0,
 * which has a log-barrier formulation
 * min. kappa * (0.5 * v.T * Q * v + b * v) - sum_log(e - G * v),
 * where sum_log refers to taking the log of every entry in a vector and then
 * summing them; kappa is the log barrier weight.
 *
 * The phase-1 program, which finds a feasible solution to the QP, is given by
 * min. s
 *  s.t. G * v - e <= s.
 * The QP is feasible if s < 0.
 */
class QpLogBarrierSolver : public LogBarrierSolver {
public:
  explicit QpLogBarrierSolver(bool use_free_solver = false);
  void SolvePhaseOne(const Eigen::Ref<const Eigen::MatrixXd> &G,
                     const Eigen::Ref<const Eigen::VectorXd> &e,
                     drake::EigenPtr<Eigen::VectorXd> v0_ptr) const override;

  /*
   * F is the log-barrier objective which we'd like to minimize.
   */
  double CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
               const Eigen::Ref<const Eigen::VectorXd> &b,
               const Eigen::Ref<const Eigen::MatrixXd> &G,
               const Eigen::Ref<const Eigen::VectorXd> &e, const double kappa,
               const Eigen::Ref<const Eigen::VectorXd> &v) const override;

  void
  CalcGradientAndHessian(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                         const Eigen::Ref<const Eigen::VectorXd> &b,
                         const Eigen::Ref<const Eigen::MatrixXd> &G,
                         const Eigen::Ref<const Eigen::VectorXd> &e,
                         const Eigen::Ref<const Eigen::VectorXd> &v,
                         double kappa, drake::EigenPtr<Eigen::VectorXd> Df_ptr,
                         drake::EigenPtr<Eigen::MatrixXd> H_ptr) const override;
};

/*
 * Consider the SOCP
 * min. 0.5 * v.T * Q * v + b.T * v
 *  s.t. -G_i * v + [e_i, 0, 0] \in Q^3,
 *  where Q^3 is the 3-dimensional second-order cone; G_i is a (3, n_v)
 *  matrix; e_i is a scalar. We concatenate G_i and e_i vertically:
 *  G := [[G_1], ...[G_n]], with shape (3 * n, n_v), and
 *  e := [e_1, ... e_n], with shape (n,).
 *
 * For convenience, we define
 * w_i := -G_i * v + [e_i, 0, 0],
 * so that the cone constraints in the SOCP can be expressed as
 * w_i[0]**2 >= w_i[1]**2 + w_i[2]**2.
 *
 * The SOCP has a log-barrier formulation
 * min. kappa * (0.5 * v.T * Q * v + b * v)
 *      - sum_log(w_i[0]**2 - w_i[1]**2 - w_i[2]**),
 * where sum_log refers to taking the log of every entry in a vector and then
 * summing them; kappa is the log barrier weight.
 */
class SocpLogBarrierSolver : public LogBarrierSolver {
public:
  explicit SocpLogBarrierSolver(bool use_free_solver = false);
  void SolvePhaseOne(const Eigen::Ref<const Eigen::MatrixXd> &G,
                     const Eigen::Ref<const Eigen::VectorXd> &e,
                     drake::EigenPtr<Eigen::VectorXd> v0_ptr) const override;

  /*
   * F is the log-barrier objective which we'd like to minimize.
   */
  double CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
               const Eigen::Ref<const Eigen::VectorXd> &b,
               const Eigen::Ref<const Eigen::MatrixXd> &G,
               const Eigen::Ref<const Eigen::VectorXd> &e, const double kappa,
               const Eigen::Ref<const Eigen::VectorXd> &v) const override;

  void
  CalcGradientAndHessian(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                         const Eigen::Ref<const Eigen::VectorXd> &b,
                         const Eigen::Ref<const Eigen::MatrixXd> &G,
                         const Eigen::Ref<const Eigen::VectorXd> &e,
                         const Eigen::Ref<const Eigen::VectorXd> &v,
                         double kappa, drake::EigenPtr<Eigen::VectorXd> Df_ptr,
                         drake::EigenPtr<Eigen::MatrixXd> H_ptr) const override;

  template <class T>
  static T DoCalcF(const Eigen::Ref<const drake::MatrixX<T>> &Q,
                   const Eigen::Ref<const drake::VectorX<T>> &b,
                   const Eigen::Ref<const drake::MatrixX<T>> &G,
                   const Eigen::Ref<const drake::VectorX<T>> &e,
                   const double kappa,
                   const Eigen::Ref<const drake::VectorX<T>> &v);

  template <class T>
  static drake::Vector3<T>
  CalcWi(const Eigen::Ref<const drake::Matrix3X<T>> &G_i, const T e_i,
         const Eigen::Ref<const drake::VectorX<T>> &v);
};

template <class T>
drake::Vector3<T>
SocpLogBarrierSolver::CalcWi(const Eigen::Ref<const drake::Matrix3X<T>> &G_i,
                             const T e_i,
                             const Eigen::Ref<const drake::VectorX<T>> &v) {
  drake::Vector3<T> w = -G_i * v;
  w[0] += e_i;
  return w;
}

template <class T>
T SocpLogBarrierSolver::DoCalcF(const Eigen::Ref<const drake::MatrixX<T>> &Q,
                                const Eigen::Ref<const drake::VectorX<T>> &b,
                                const Eigen::Ref<const drake::MatrixX<T>> &G,
                                const Eigen::Ref<const drake::VectorX<T>> &e,
                                const double kappa,
                                const Eigen::Ref<const drake::VectorX<T>> &v) {
  const int n_c = G.rows() / 3;
  const int n_v = G.cols();

  T output = 0.5 * v.dot(Q * v) + v.dot(b);
  output *= kappa;

  for (int i = 0; i < n_c; i++) {
    drake::Vector3<T> w = CalcWi<T>(G.block(i * 3, 0, 3, n_v), e[i], v);
    const T d = -w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if (d > 0 or w[0] < 0) {
      return {std::numeric_limits<double>::infinity()};
    }
    using Eigen::log; // AutoDiffXd
    using std::log;   // double
    output += -log(-d);
  }
  return output;
}
