#pragma once

#include "qsim/batch_quasistatic_simulator.h"
#include "qsim/quasistatic_simulator.h"

class FiniteDiffGradientCalculatorBase {
 public:
  FiniteDiffGradientCalculatorBase(int n_x, int n_u);
  virtual Eigen::MatrixXd CalcB(
      const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
      const Eigen::Ref<const Eigen::VectorXd>& u_nominal, double du,
      QuasistaticSimParameters sim_params) = 0;

  virtual Eigen::MatrixXd CalcA(
      const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
      const Eigen::Ref<const Eigen::VectorXd>& u_nominal, double dx,
      QuasistaticSimParameters sim_params) = 0;

 protected:
  const int n_x_{0};
  const int n_u_{0};
};

class FiniteDiffGradientCalculator final
    : public FiniteDiffGradientCalculatorBase {
 public:
  explicit FiniteDiffGradientCalculator(QuasistaticSimulator* q_sim);

  Eigen::MatrixXd CalcB(const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
                        const Eigen::Ref<const Eigen::VectorXd>& u_nominal,
                        double du,
                        QuasistaticSimParameters sim_params) override;

  Eigen::MatrixXd CalcA(const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
                        const Eigen::Ref<const Eigen::VectorXd>& u_nominal,
                        double dx,
                        QuasistaticSimParameters sim_params) override;

 private:
  QuasistaticSimulator* q_sim_;
};

class BatchFiniteDiffGradientCalculator final
    : public FiniteDiffGradientCalculatorBase {
 public:
  explicit BatchFiniteDiffGradientCalculator(
      BatchQuasistaticSimulator* q_sim_batch);

  Eigen::MatrixXd CalcB(const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
                        const Eigen::Ref<const Eigen::VectorXd>& u_nominal,
                        double du,
                        QuasistaticSimParameters sim_params) override;

  Eigen::MatrixXd CalcA(const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
                        const Eigen::Ref<const Eigen::VectorXd>& u_nominal,
                        double dx,
                        QuasistaticSimParameters sim_params) override;

 private:
  BatchQuasistaticSimulator* q_sim_batch_;
  Eigen::MatrixXd x_buffer_A_;
  Eigen::MatrixXd u_buffer_A_;
  Eigen::MatrixXd x_buffer_B_;
  Eigen::MatrixXd u_buffer_B_;
};
