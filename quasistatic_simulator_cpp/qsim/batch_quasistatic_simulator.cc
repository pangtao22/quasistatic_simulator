#include "qsim/batch_quasistatic_simulator.h"

#include <spdlog/spdlog.h>

#include <future>
#include <queue>
#include <stack>

#include "qsim/quasistatic_simulator.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

BatchQuasistaticSimulator::BatchQuasistaticSimulator(
    const std::string& model_directive_path,
    const std::unordered_map<std::string, Eigen::VectorXd>& robot_stiffness_str,
    const std::unordered_map<std::string, std::string>& object_sdf_paths,
    const QuasistaticSimParameters& sim_params)
    : BatchQuasistaticSimulator(*QuasistaticSimulator::MakeQuasistaticSimulator(
          model_directive_path, robot_stiffness_str, object_sdf_paths,
          sim_params)) {}

BatchQuasistaticSimulator::BatchQuasistaticSimulator(
    const QuasistaticSimulator& q_sim)
    : num_max_parallel_executions(std::thread::hardware_concurrency()),
      solver_(std::make_unique<drake::solvers::GurobiSolver>()) {
  std::random_device rd;
  gen_.seed(rd());
  for (int i = 0; i < num_max_parallel_executions; i++) {
    q_sims_.push_back(std::move(q_sim.Clone()));
  }
}

Eigen::MatrixXd BatchQuasistaticSimulator::CalcBundledB(
    QuasistaticSimulator* q_sim, const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::MatrixXd>& du,
    const QuasistaticSimParameters& sim_params) {
  const auto n_q = q.size();
  const auto n_u = u.size();
  MatrixXd B_bundled(n_q, n_u);
  B_bundled.setZero();
  const auto& sp = q_sim->get_sim_params();

  int n_valid = 0;
  for (int i = 0; i < du.rows(); i++) {
    VectorXd u_new = u + du.row(i).transpose();
    try {
      QuasistaticSimulator::CalcDynamics(q_sim, q, u_new, sim_params);
      B_bundled += q_sim->get_Dq_nextDqa_cmd();
      n_valid++;
    } catch (std::runtime_error& err) {
      spdlog::warn(err.what());
    }
  }
  B_bundled /= n_valid;

  return B_bundled;
}

std::tuple<bool, bool> IsABNeeded(GradientMode gm) {
  bool calc_A = gm == GradientMode::kAB;
  bool calc_B = calc_A || gm == GradientMode::kBOnly;
  return {calc_A, calc_B};
}

std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>,
           std::vector<Eigen::MatrixXd>, std::vector<bool>>
BatchQuasistaticSimulator::CalcDynamicsSerial(
    const Eigen::Ref<const Eigen::MatrixXd>& x_batch,
    const Eigen::Ref<const Eigen::MatrixXd>& u_batch,
    const QuasistaticSimParameters& sim_params) const {
  const auto& [calc_A, calc_B] = IsABNeeded(sim_params.gradient_mode);

  auto& q_sim = get_q_sim();
  const size_t n_tasks = x_batch.rows();
  DRAKE_THROW_UNLESS(n_tasks == u_batch.rows());
  MatrixXd x_next_batch(x_batch);
  std::vector<MatrixXd> A_batch;
  std::vector<MatrixXd> B_batch;
  std::vector<bool> is_valid_batch(n_tasks);
  const auto n_q = x_batch.cols();
  const auto n_u = u_batch.cols();

  for (int i = 0; i < n_tasks; i++) {
    try {
      x_next_batch.row(i) = QuasistaticSimulator::CalcDynamics(
          &q_sim, x_batch.row(i), u_batch.row(i), sim_params);
      if (calc_B) {
        B_batch.emplace_back(q_sim.get_Dq_nextDqa_cmd());
      }
      if (calc_A) {
        A_batch.emplace_back(q_sim.get_Dq_nextDq());
      }
      is_valid_batch[i] = true;
    } catch (std::runtime_error& err) {
      is_valid_batch[i] = false;
    }
  }

  return {x_next_batch, A_batch, B_batch, is_valid_batch};
}

std::vector<size_t> BatchQuasistaticSimulator::CalcBatchSizes(
    size_t n_tasks, size_t n_threads) {
  const auto batch_size = n_tasks / n_threads;
  std::vector<size_t> batch_sizes(n_threads, batch_size);

  const auto n_leftovers = n_tasks - n_threads * batch_size;
  for (int i = 0; i < n_leftovers; i++) {
    batch_sizes[i] += 1;
  }

  return batch_sizes;
}

template <typename T>
std::vector<T> AssembleListsOfVectors(std::list<std::vector<T>>* list_of_vec) {
  std::vector<T> batch;
  for (auto iter = list_of_vec->begin(); iter != list_of_vec->end(); iter++) {
    batch.insert(batch.end(), std::make_move_iterator(iter->begin()),
                 std::make_move_iterator(iter->end()));
  }
  return batch;
}

std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>,
           std::vector<Eigen::MatrixXd>, std::vector<bool>>
BatchQuasistaticSimulator::CalcDynamicsParallel(
    const Eigen::Ref<const Eigen::MatrixXd>& x_batch,
    const Eigen::Ref<const Eigen::MatrixXd>& u_batch,
    const QuasistaticSimParameters& sim_params) const {
  const auto [calc_A, calc_B] = IsABNeeded(sim_params.gradient_mode);
  auto& q_sim = get_q_sim();

  // Compute number of threads and batch size for each thread.
  const size_t n_tasks = x_batch.rows();
  DRAKE_THROW_UNLESS(n_tasks == u_batch.rows());
  const auto n_threads = std::min(num_max_parallel_executions, n_tasks);
  const auto batch_sizes = CalcBatchSizes(n_tasks, n_threads);

  // Allocate storage for results.
  const auto n_q = x_batch.cols();
  const auto n_u = u_batch.cols();

  std::list<MatrixXd> x_next_list;
  std::list<std::vector<MatrixXd>> A_list;
  std::list<std::vector<MatrixXd>> B_list;
  std::list<std::vector<bool>> is_valid_list;
  for (int i = 0; i < n_threads; i++) {
    x_next_list.emplace_back(batch_sizes[i], n_q);

    if (calc_B) {
      B_list.emplace_back(batch_sizes[i]);
    }

    if (calc_A) {
      A_list.emplace_back(batch_sizes[i]);
    }

    is_valid_list.emplace_back(batch_sizes[i]);
  }

  // Launch threads.
  std::list<std::future<void>> operations;
  auto q_sim_iter = q_sims_.begin();
  auto x_next_iter = x_next_list.begin();
  auto A_iter = A_list.begin();
  auto B_iter = B_list.begin();
  auto is_valid_iter = is_valid_list.begin();

  for (int i_thread = 0; i_thread < n_threads; i_thread++) {
    // subscript _t indicates a quantity for a thread.
    auto calc_dynamics_batch = [&q_sim = *q_sim_iter, &x_batch, &u_batch,
                                &x_next_t = *x_next_iter, &A_t = *A_iter,
                                &B_t = *B_iter, &is_valid_t = *is_valid_iter,
                                &batch_sizes, i_thread, &sim_params,
                                calc_A = calc_A, calc_B = calc_B] {
      const auto offset = std::accumulate(batch_sizes.begin(),
                                          batch_sizes.begin() + i_thread, 0);
      for (int i = 0; i < batch_sizes[i_thread]; i++) {
        try {
          x_next_t.row(i) = QuasistaticSimulator::CalcDynamics(
              q_sim.get(), x_batch.row(i + offset), u_batch.row(i + offset),
              sim_params);

          if (calc_B) {
            B_t[i] = q_sim->get_Dq_nextDqa_cmd();
          }

          if (calc_A) {
            A_t[i] = q_sim->get_Dq_nextDq();
          }

          is_valid_t[i] = true;
        } catch (std::runtime_error& err) {
          is_valid_t[i] = false;
          spdlog::warn(err.what());
        }
      }
    };

    operations.emplace_back(
        std::async(std::launch::async, std::move(calc_dynamics_batch)));

    q_sim_iter++;
    x_next_iter++;
    is_valid_iter++;
    if (calc_A) {
      A_iter++;
    }
    if (calc_B) {
      B_iter++;
    }
  }

  // Collect results from threads, and assemble x_next_batch.
  // x_next;
  int i_thread = 0;
  int i_start = 0;
  MatrixXd x_next_batch(n_tasks, n_q);
  x_next_iter = x_next_list.begin();
  for (auto& op : operations) {
    op.get();  // catch exceptions.
    auto batch_size = batch_sizes[i_thread];
    x_next_batch.block(i_start, 0, batch_size, n_q) = *x_next_iter;

    i_start += batch_size;
    i_thread++;
    x_next_iter++;
  }

  auto A_batch = AssembleListsOfVectors(&A_list);
  auto B_batch = AssembleListsOfVectors(&B_list);
  auto is_valid_batch = AssembleListsOfVectors(&is_valid_list);
  return {x_next_batch, A_batch, B_batch, is_valid_batch};
}

Eigen::MatrixXd BatchQuasistaticSimulator::SampleGaussianMatrix(
    int n_rows, const Eigen::Ref<const Eigen::VectorXd>& mu,
    const Eigen::Ref<const Eigen::VectorXd>& std) const {
  DRAKE_THROW_UNLESS(mu.size() == std.size());
  Eigen::MatrixXd A(n_rows, std.size());
  for (int i = 0; i < std.size(); i++) {
    std::normal_distribution<> d{mu[i], std[i]};
    A.col(i) = MatrixXd::NullaryExpr(n_rows, 1, [&]() {
      return d(gen_);
    });
  }

  return A;
}

std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>,
           std::vector<Eigen::VectorXd>>
BatchQuasistaticSimulator::CalcBundledABcTrjScalarStd(
    const Eigen::Ref<const Eigen::MatrixXd>& x_trj,
    const Eigen::Ref<const Eigen::MatrixXd>& u_trj, double std_u,
    const QuasistaticSimParameters& sim_params, int n_samples,
    std::optional<int> seed) const {
  auto std_u_vec = VectorXd::Constant(u_trj.cols(), std_u);
  return CalcBundledABcTrj(x_trj, u_trj, std_u_vec, sim_params, n_samples,
                           seed);
}

/*
 * The input variable "samples" consists of (T * n_samples) samples of type
 *  M, which can be Eigen::MatrixXd or Eigen::VectorXd.
 *
 * Returns a list of length T, where each element is the average of n_samples
 *  elements in the input variable samples.
 */
template <typename M>
std::vector<M> CalcBundledFromSamples(const std::vector<M>& samples,
                                      const std::vector<bool>& is_sample_valid,
                                      const int T, const int n_samples) {
  DRAKE_THROW_UNLESS(samples.size() == T * n_samples);
  std::vector<M> bundled;
  for (int t = 0; t < T; t++) {
    int i_start = t * n_samples;
    int n_valid_samples = 0;

    bundled.emplace_back(samples[0]);
    bundled.back().setZero();

    for (int i = 0; i < n_samples; i++) {
      if (is_sample_valid[i_start + i]) {
        n_valid_samples++;
        bundled.back() += samples[i_start + i];
      }
    }

    bundled.back() /= n_valid_samples;
  }

  return bundled;
}

std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>,
           std::vector<Eigen::VectorXd>>
BatchQuasistaticSimulator::CalcBundledABcTrj(
    const Eigen::Ref<const Eigen::MatrixXd>& x_trj,
    const Eigen::Ref<const Eigen::MatrixXd>& u_trj,
    const Eigen::Ref<const Eigen::VectorXd>& std_u,
    const QuasistaticSimParameters& sim_params, int n_samples,
    std::optional<int> seed) const {
  if (seed.has_value()) {
    gen_.seed(seed.value());
  }

  const int T = u_trj.rows();
  DRAKE_ASSERT(x_trj.rows() == T);

  const int n_x = x_trj.cols();
  const int n_u = u_trj.cols();

  MatrixXd x_batch(T * n_samples, n_x);
  MatrixXd u_batch(T * n_samples, n_u);

  for (int t = 0; t < T; t++) {
    int i_start = t * n_samples;
    auto u_batch_t = SampleGaussianMatrix(n_samples, u_trj.row(t), std_u);
    x_batch(Eigen::seqN(i_start, n_samples), Eigen::all).rowwise() =
        x_trj.row(t);
    u_batch(Eigen::seqN(i_start, n_samples), Eigen::all) = u_batch_t;
  }

  auto [x_next_batch, A_batch, B_batch, is_valid_batch] =
      CalcDynamicsParallel(x_batch, u_batch, sim_params);

  std::vector<VectorXd> c_bundled;
  for (int t = 0; t < T; t++) {
    int i_start = t * n_samples;
    int n_valid_samples = 0;
    c_bundled.emplace_back(n_x);
    c_bundled.back().setZero();

    for (int i = 0; i < n_samples; i++) {
      if (is_valid_batch[i_start + i]) {
        n_valid_samples++;
        c_bundled.back() += x_next_batch.row(i_start + i);
      }
    }

    c_bundled.back() /= n_valid_samples;
  }

  std::vector<MatrixXd> A_bundled, B_bundled;
  if (!A_batch.empty()) {
    A_bundled = CalcBundledFromSamples(A_batch, is_valid_batch, T, n_samples);
  }
  if (!B_batch.empty()) {
    B_bundled = CalcBundledFromSamples(B_batch, is_valid_batch, T, n_samples);
  }

  return {A_bundled, B_bundled, c_bundled};
}

template <typename T>
bool IsFutureReady(const std::future<T>& future) {
  // future.wait_for() is the only method to check the status of a future
  // without waiting for it to complete.
  const std::future_status status =
      future.wait_for(std::chrono::milliseconds(1));
  return (status == std::future_status::ready);
}

std::stack<int> BatchQuasistaticSimulator::InitializeSimulatorStack() const {
  std::stack<int> sims;
  for (int i = 0; i < num_max_parallel_executions; i++) {
    sims.push(i);
  }
  return sims;
}

std::vector<Eigen::MatrixXd> BatchQuasistaticSimulator::CalcBundledBTrjDirect(
    const Eigen::Ref<const Eigen::MatrixXd>& x_trj,
    const Eigen::Ref<const Eigen::MatrixXd>& u_trj, double std_u,
    QuasistaticSimParameters sim_params, int n_samples,
    std::optional<int> seed) const {
  sim_params.gradient_mode = GradientMode::kBOnly;
  if (seed.has_value()) {
    gen_.seed(seed.value());
  }

  // Determine the number of threads.
  const size_t T = u_trj.rows();
  DRAKE_THROW_UNLESS(x_trj.rows() == T);
  const auto n_threads = std::min(num_max_parallel_executions, T);

  // Allocate storage for results.
  const auto n_q = x_trj.cols();
  const auto n_u = u_trj.cols();
  std::vector<MatrixXd> B_batch(T, MatrixXd::Zero(n_q, n_u));

  // Generate samples.
  std::vector<MatrixXd> du_trj(T);
  std::normal_distribution<> d{0, std_u};
  for (int t = 0; t < T; t++) {
    du_trj[t] = MatrixXd::NullaryExpr(n_samples, n_u, [&]() {
      return d(gen_);
    });
  }

  // Storage for active parallel simulation operations.
  std::list<std::future<int>> active_operations;
  int n_bundled_B_dispatched = 0;
  auto available_sims = InitializeSimulatorStack();

  while (!active_operations.empty() || n_bundled_B_dispatched < T) {
    // Check for completed operations.
    for (auto op = active_operations.begin(); op != active_operations.end();) {
      if (IsFutureReady(*op)) {
        auto sim_idx = op->get();
        op = active_operations.erase(op);
        available_sims.push(sim_idx);
      } else {
        ++op;
      }
    }

    // Dispatch new operations.
    while (static_cast<int>(active_operations.size()) < n_threads &&
           n_bundled_B_dispatched < T) {
      DRAKE_THROW_UNLESS(!available_sims.empty());
      auto idx_sim = available_sims.top();
      available_sims.pop();

      auto calc_B_bundled =
          [&q_sim = q_sims_[idx_sim], &x_trj = std::as_const(x_trj),
           &u_trj = std::as_const(u_trj), &du_trj = std::as_const(du_trj),
           &B_batch, t = n_bundled_B_dispatched, sim_params, idx_sim] {
            B_batch[t] = CalcBundledB(q_sim.get(), x_trj.row(t), u_trj.row(t),
                                      du_trj[t], sim_params);
            return idx_sim;
          };

      active_operations.emplace_back(
          std::async(std::launch::async, std::move(calc_B_bundled)));
      ++n_bundled_B_dispatched;
    }

    // Wait a bit before checking for completion.
    // For the planar hand and ball system, computing forward dynamics and
    // its gradient takes a bit more than 1ms on average.
    std::this_thread::sleep_for(std::chrono::milliseconds(n_samples));
  }

  return B_batch;
}

// TODO(pang?): This generates all 0s for unactuated DOFs for some reason...
// TLDR: it doesn't work at the moment. Use the python version instead of this!
std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
BatchQuasistaticSimulator::CalcBcLstsq(
    const Eigen::Ref<const Eigen::VectorXd>& x_nominal,
    const Eigen::Ref<const Eigen::VectorXd>& u_nominal,
    QuasistaticSimParameters sim_params,
    const Eigen::Ref<const Eigen::VectorXd>& u_std, int n_samples) const {
  const auto n_x = x_nominal.size();
  const auto n_u = u_nominal.size();
  MatrixXd x_batch(n_samples, n_x);
  x_batch.rowwise() = x_nominal.transpose();
  const auto u_batch = SampleGaussianMatrix(n_samples, u_nominal, u_std);
  sim_params.gradient_mode = GradientMode::kNone;
  auto [x_next_batch, A_Batch, B_batch, is_valid_batch] =
      CalcDynamicsParallel(x_batch, u_batch, sim_params);

  // Extract valid samples.
  const auto n_valid =
      std::accumulate(is_valid_batch.begin(), is_valid_batch.end(), 0);
  if (n_valid == 0) {
    throw std::runtime_error("No valid dynamics samples.");
  }

  VectorXd x_next_mean(n_x);
  x_next_mean.setZero();
  std::vector<int> idx_valid;
  for (int i = 0; i < n_samples; i++) {
    if (!is_valid_batch[i]) {
      continue;
    }
    x_next_mean += x_next_batch.row(i) / n_valid;
    idx_valid.push_back(i);
  }

  // Data centering.
  MatrixXd dx =
      x_next_batch(idx_valid, Eigen::all).rowwise() - x_next_mean.transpose();
  MatrixXd du =
      u_batch(idx_valid, Eigen::all).rowwise() - u_nominal.transpose();

  auto prog = drake::solvers::MathematicalProgram();
  auto B = prog.NewContinuousVariables(n_x, n_u, "B");

  // Column-wise costs.
  for (int j = 0; j < n_u; j++) {
    prog.Add2NormSquaredCost(du, dx.col(j), B.transpose().col(j));
  }

  auto result = drake::solvers::MathematicalProgramResult();
  solver_->Solve(prog, {}, {}, &result);

  if (!result.is_success()) {
    throw std::runtime_error("Failed to solve for B using Least squares.");
  }

  return {result.GetSolution(B), x_next_mean};
}
