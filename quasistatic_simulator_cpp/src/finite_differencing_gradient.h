#pragma once

#include "quasistatic_simulator.h"

class FiniteDiffGradientCalculator {
public:
  explicit FiniteDiffGradientCalculator(QuasistaticSimulator &q_sim)
      : q_sim_(&q_sim) {};
  /*
   * FD stands for Finite Differencing.
   */
  Eigen::MatrixXd CalcB(const Eigen::Ref<const Eigen::VectorXd> &x_nominal,
                        const Eigen::Ref<const Eigen::VectorXd> &u_nominal,
                        const double du, QuasistaticSimParameters sim_params);

  Eigen::MatrixXd CalcA(const Eigen::Ref<const Eigen::VectorXd> &x_nominal,
                        const Eigen::Ref<const Eigen::VectorXd> &u_nominal,
                        const double dx, QuasistaticSimParameters sim_params);

private:
  QuasistaticSimulator *q_sim_;
};
