#include "quasistatic_simulator.h"
#include <list>
#include <tuple>

class BatchQuasistaticSimulator {
public:
  BatchQuasistaticSimulator(
      const std::string &model_directive_path,
      const std::unordered_map<std::string, Eigen::VectorXd>
          &robot_stiffness_str,
      const std::unordered_map<std::string, std::string> &object_sdf_paths,
      const QuasistaticSimParameters &sim_params);

  /*
   * Each row in x_batch and u_batch represent a pair of current states and
   * inputs. The function returns a tuple of
   *  (x_next_batch, A_batch, B_batch, is_valid_batch), where
   *  - x_next_batch.row(i) = f(x_batch.row(i), u_batch.row(i))
   *  - A_batch[i], B_batch[i] are the bundled linearized dynamics, as in
   *    x_next = A * x + B * u.
   *  - is_valid_batch[i] is false if the forward dynamics fails to solve or
   *    B_batch[i] is nan, which can happen if the least square solve during
   *    the application of implicit function theorem to the KKT condition of
   *    the QP fails.
   *
   *  Behaviors under different gradient_mode:
   *    kNone: A_Batch, B_batch has 0 length.
   *    kAB: A_batch[i] is a (n_q, n_q) matrix,
   *         B_batch[i] is a (n_q, n_a) matrix.
   *    kBOnly: A_Batch has 0 length, B_batch[i] is a (n_q, n_a) matrix.
   */
  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>,
             std::vector<Eigen::MatrixXd>, std::vector<bool>>
  CalcDynamicsParallel(const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
                       const Eigen::Ref<const Eigen::MatrixXd> &u_batch,
                       const QuasistaticSimParameters &sim_params) const;

  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>,
             std::vector<Eigen::MatrixXd>, std::vector<bool>>
  CalcDynamicsSerial(const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
                     const Eigen::Ref<const Eigen::MatrixXd> &u_batch,
                     const QuasistaticSimParameters &sim_params) const;

  /*
   * Minimizes the least square error of
   * x_next_batch - x_next_batch_mean = (u_batch - u_nominal) * B.transpose().
   */
  std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
  CalcBcLstsq(
      const Eigen::Ref<const Eigen::VectorXd> &x_nominal,
      const Eigen::Ref<const Eigen::VectorXd> &u_nominal,
      QuasistaticSimParameters sim_params,
      const Eigen::Ref<const Eigen::VectorXd> &u_std,
      int n_samples) const;

  /*
   * x_trj: (T, dim_x)
   * u_trj: (T, dim_u)
   *
   * In the tuple: (A_list of length T or 0,
   *                B_list of length T or 0,
   *                c_list of length T)
   * A_list and B_list can be empty, depending on sim_params.gradient_mode.
   */
  std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>,
             std::vector<Eigen::VectorXd>>
  CalcBundledABcTrj(const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
                    const Eigen::Ref<const Eigen::MatrixXd> &u_trj,
                    const Eigen::Ref<const Eigen::VectorXd> &std_u,
                    const QuasistaticSimParameters &sim_params, int n_samples,
                    std::optional<int> seed) const;

  std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>,
             std::vector<Eigen::VectorXd>>
  CalcBundledABcTrjScalarStd(const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
                             const Eigen::Ref<const Eigen::MatrixXd> &u_trj,
                             double std_u,
                             const QuasistaticSimParameters &sim_params,
                             int n_samples, std::optional<int> seed) const;

  /*
   * Implements multi-threaded computation of bundled gradient based on drake's
   * Monte-Carlo simulation:
   * https://github.com/RobotLocomotion/drake/blob/5316536420413b51871ceb4b9c1f77aedd559f71/systems/analysis/monte_carlo.cc#L42
   * But this implementation does not seem to be faster than
   * CalcBundledBTrjScalarStd, which is again  slower than the original
   * ZMQ-based PUSH-PULL scheme.
   *
   * It is a sad conclusion after almost two weeks of effort ¯\_(ツ)_/¯.
   * Well, at least I learned more about C++ and saw quite a bit of San
   * Francisco :)
   */
  std::vector<Eigen::MatrixXd>
  CalcBundledBTrjDirect(const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
                        const Eigen::Ref<const Eigen::MatrixXd> &u_trj,
                        double std_u, QuasistaticSimParameters sim_params,
                        int n_samples, std::optional<int> seed) const;

  static Eigen::MatrixXd
  CalcBundledB(QuasistaticSimulator *q_sim,
               const Eigen::Ref<const Eigen::VectorXd> &q,
               const Eigen::Ref<const Eigen::VectorXd> &u,
               const Eigen::Ref<const Eigen::MatrixXd> &du,
               const QuasistaticSimParameters &sim_params);

  Eigen::MatrixXd
  SampleGaussianMatrix(int n_rows, const Eigen::Ref<const Eigen::VectorXd> &mu,
                       const Eigen::Ref<const Eigen::VectorXd> &std) const;

  size_t get_num_max_parallel_executions() const {
    return num_max_parallel_executions;
  };
  void set_num_max_parallel_executions(size_t n) {
    num_max_parallel_executions = n;
  };

  QuasistaticSimulator &get_q_sim() const { return *q_sims_.begin(); };

private:
  static std::vector<size_t> CalcBatchSizes(size_t n_tasks, size_t n_threads);

  std::stack<int> InitializeSimulatorStack() const;
  size_t num_max_parallel_executions{0};

  std::unique_ptr<drake::solvers::GurobiSolver> solver_;

  mutable std::vector<QuasistaticSimulator> q_sims_;
  mutable std::mt19937 gen_;
};
