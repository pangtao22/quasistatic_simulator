#include <iostream>
#include <random>
#include <thread>

#include <gtest/gtest.h>

#include "get_model_paths.h"
#include "quasistatic_parser.h"
#include "test_utilities.h"

using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;

MatrixXd CreateRandomMatrix(int n_rows, int n_cols, std::mt19937 &gen) {
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  return MatrixXd::NullaryExpr(n_rows, n_cols, [&]() { return dis(gen); });
}

class TestBatchQuasistaticSimulator : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    // Make sure that n_tasks_ is not divisible by hardware_concurrency.
    n_tasks_ = std::thread::hardware_concurrency() * 2 + 1;
    sim_params_ = {};
    sim_params_.use_free_solvers = GetParam();
  }

  // TODO: simplify these setup functions with QuasistaticParser.
  void SetUpPlanarHand() {
    const string kQModelPath =
        GetQsimModelsPath() / "q_sys" / "planar_hand_ball.yml";
    auto parser = QuasistaticParser(kQModelPath);

    sim_params_.h = h_;
    sim_params_.gravity = Vector3d(0, 0, -10);
    sim_params_.nd_per_contact = 2;
    sim_params_.contact_detection_tolerance = 1.0;
    sim_params_.is_quasi_dynamic = true;

    parser.set_sim_params(sim_params_);
    q_sim_batch_ = parser.MakeBatchSimulator();

    const string robot_l_name("arm_left");
    const string robot_r_name("arm_right");
    const string object_name("sphere");

    auto &q_sim = q_sim_batch_->get_q_sim();
    const auto name_to_idx_map = q_sim.GetModelInstanceNameToIndexMap();
    const auto idx_l = name_to_idx_map.at(robot_l_name);
    const auto idx_r = name_to_idx_map.at(robot_r_name);
    const auto idx_o = name_to_idx_map.at(object_name);

    ModelInstanceIndexToVecMap q0_dict = {{idx_o, Vector3d(0, 0.316, 0)},
                                          {idx_l, Vector2d(-0.775, -0.785)},
                                          {idx_r, Vector2d(0.775, 0.785)}};

    VectorXd q0 = q_sim.GetQVecFromDict(q0_dict);
    VectorXd u0 = q_sim.GetQaCmdVecFromDict(q0_dict);

    SampleUBatch(u0, 0.1);
    SetXBatch(q0);
  }

  void SetUpAllegroHand() {
    const string kQModelPath =
        GetQsimModelsPath() / "q_sys" / "allegro_hand_and_sphere.yml";
    auto parser = QuasistaticParser(kQModelPath);
    sim_params_.h = h_;
    sim_params_.gravity = Vector3d(0, 0, 0);
    sim_params_.nd_per_contact = 4;
    sim_params_.contact_detection_tolerance = 0.025;
    sim_params_.is_quasi_dynamic = true;
    sim_params_.log_barrier_weight = 100;
    parser.set_sim_params(sim_params_);
    q_sim_batch_ = parser.MakeBatchSimulator();

    auto &q_sim = q_sim_batch_->get_q_sim();
    const auto name_to_idx_map = q_sim.GetModelInstanceNameToIndexMap();
    const string robot_name("allegro_hand_right");
    const string object_name("sphere");
    const auto idx_r = name_to_idx_map.at(robot_name);
    const auto idx_o = name_to_idx_map.at(object_name);

    VectorXd q_u0(7);
    q_u0 << 1, 0, 0, 0, -0.081, 0.001, 0.071;

    VectorXd q_a0(q_sim.num_actuated_dofs());
    q_a0 << 0.035, 0.753, 0.741, 0.833, -0.144, 0.745, 0.619, 0.701, -0.069,
        0.785, 0.829, 0.904, 0.633, 1.024, 0.641, 0.824;

    ModelInstanceIndexToVecMap q0_dict = {{idx_o, q_u0}, {idx_r, q_a0}};

    VectorXd q0 = q_sim.GetQVecFromDict(q0_dict);
    VectorXd u0 = q_sim.GetQaCmdVecFromDict(q0_dict);

    SampleUBatch(u0, 0.1);
    SetXBatch(q0);
  };

  void SampleUBatch(const Eigen::Ref<const Eigen::VectorXd> &u0,
                    double interval_size) {
    std::mt19937 gen(1);
    u_batch_ = interval_size * CreateRandomMatrix(n_tasks_, u0.size(), gen);
    u_batch_.rowwise() += u0.transpose();
  }

  void SetXBatch(const Eigen::Ref<const Eigen::VectorXd> &x0) {
    x_batch_.resize(n_tasks_, x0.size());
    x_batch_.setZero();
    x_batch_.rowwise() += x0.transpose();
  }

  void CompareIsValid(const std::vector<bool> &is_valid_batch_1,
                      const std::vector<bool> &is_valid_batch_2) const {
    EXPECT_EQ(n_tasks_, is_valid_batch_1.size());
    EXPECT_EQ(n_tasks_, is_valid_batch_2.size());
    for (int i = 0; i < is_valid_batch_1.size(); i++) {
      EXPECT_EQ(is_valid_batch_1[i], is_valid_batch_2[i]);
    }
  }

  void CompareXNext(const Eigen::Ref<const MatrixXd> &x_next_batch_1,
                    const Eigen::Ref<const MatrixXd> &x_next_batch_2,
                    const double tol = 1e-6) const {
    EXPECT_EQ(n_tasks_, x_next_batch_1.rows());
    EXPECT_EQ(n_tasks_, x_next_batch_2.rows());
    const double avg_diff =
        (x_next_batch_2 - x_next_batch_1).matrix().rowwise().norm().sum() /
        n_tasks_;
    EXPECT_LT(avg_diff, tol);
  }

  void CompareMatrices(const std::vector<MatrixXd> &M_batch_1,
                       const std::vector<MatrixXd> &M_batch_2,
                       const double tol) const {
    EXPECT_EQ(n_tasks_, M_batch_1.size());
    EXPECT_EQ(n_tasks_, M_batch_2.size());
    for (int i = 0; i < n_tasks_; i++) {
      double err = (M_batch_1[i] - M_batch_2[i]).norm();
      double rel_err = err / M_batch_1[i].norm();
      EXPECT_LT(err, tol);
      EXPECT_LT(rel_err, 0.01);
    }
  }

  int n_tasks_{0};
  const double h_{0.1};
  QuasistaticSimParameters sim_params_;
  MatrixXd u_batch_, x_batch_;
  std::unique_ptr<BatchQuasistaticSimulator> q_sim_batch_;
};

TEST_P(TestBatchQuasistaticSimulator, TestForwardDynamicsPlanarHand) {
  SetUpPlanarHand();
  auto [x_next_batch_parallel, A_batch_parallel, B_batch_parallel,
        is_valid_batch_parallel] =
      q_sim_batch_->CalcDynamicsParallel(x_batch_, u_batch_, sim_params_);

  auto [x_next_batch_serial, A_batch_serial, B_batch_serial,
        is_valid_batch_serial] =
      q_sim_batch_->CalcDynamicsSerial(x_batch_, u_batch_, sim_params_);
  // is_valid.
  CompareIsValid(is_valid_batch_parallel, is_valid_batch_serial);

  // x_next.
  CompareXNext(x_next_batch_parallel, x_next_batch_serial);

  // B.
  EXPECT_EQ(B_batch_parallel.size(), 0);
  EXPECT_EQ(B_batch_serial.size(), 0);

  // A.
  EXPECT_EQ(A_batch_parallel.size(), 0);
  EXPECT_EQ(B_batch_serial.size(), 0);
}

TEST_P(TestBatchQuasistaticSimulator, TestForwardDynamicsAllegroHand) {
  SetUpAllegroHand();
  std::vector<ForwardDynamicsMode> forward_modes_to_test = {
      ForwardDynamicsMode::kQpMp, ForwardDynamicsMode::kSocpMp,
      ForwardDynamicsMode::kLogPyramidMp, ForwardDynamicsMode::kLogPyramidMy,
      ForwardDynamicsMode::kLogIcecream};
  std::vector<double> tol = {1e-6, 1e-6, 1e-6, 1e-5, 1e-5};
  if (sim_params_.use_free_solvers) {
    tol = {1e-4, 1e-4, 1e-4, 1e-4, 1e-4};
  }

  int i = 0;
  for (const auto forward_mode : forward_modes_to_test) {
    sim_params_.forward_mode = forward_mode;
    auto [x_next_batch_parallel, A_batch_parallel, B_batch_parallel,
          is_valid_batch_parallel] =
        q_sim_batch_->CalcDynamicsParallel(x_batch_, u_batch_, sim_params_);

    auto [x_next_batch_serial, A_batch_serial, B_batch_serial,
          is_valid_batch_serial] =
        q_sim_batch_->CalcDynamicsSerial(x_batch_, u_batch_, sim_params_);
    // is_valid.
    CompareIsValid(is_valid_batch_parallel, is_valid_batch_serial);

    // x_next.
    CompareXNext(x_next_batch_parallel, x_next_batch_serial, tol[i]);

    // B.
    EXPECT_EQ(B_batch_parallel.size(), 0);
    EXPECT_EQ(B_batch_serial.size(), 0);

    // A.
    EXPECT_EQ(A_batch_parallel.size(), 0);
    EXPECT_EQ(B_batch_serial.size(), 0);
    i++;
  }
}

TEST_P(TestBatchQuasistaticSimulator, TestGradientPlanarHand) {
  SetUpPlanarHand();
  sim_params_.gradient_mode = GradientMode::kAB;

  auto [x_next_batch_parallel, A_batch_parallel, B_batch_parallel,
        is_valid_batch_parallel] =
      q_sim_batch_->CalcDynamicsParallel(x_batch_, u_batch_, sim_params_);

  auto [x_next_batch_serial, A_batch_serial, B_batch_serial,
        is_valid_batch_serial] =
      q_sim_batch_->CalcDynamicsSerial(x_batch_, u_batch_, sim_params_);

  // is_valid.
  CompareIsValid(is_valid_batch_parallel, is_valid_batch_serial);

  // x_next.
  CompareXNext(x_next_batch_parallel, x_next_batch_serial);

  // B.
  CompareMatrices(B_batch_parallel, B_batch_serial, 1e-6);

  // A.
  CompareMatrices(A_batch_parallel, A_batch_serial, 5e-5);
}

TEST_P(TestBatchQuasistaticSimulator, TestGradientAllegroHand) {
  SetUpAllegroHand();
  sim_params_.gradient_mode = GradientMode::kAB;

  auto [x_next_batch_parallel, A_batch_parallel, B_batch_parallel,
        is_valid_batch_parallel] =
      q_sim_batch_->CalcDynamicsParallel(x_batch_, u_batch_, sim_params_);

  auto [x_next_batch_serial, A_batch_serial, B_batch_serial,
        is_valid_batch_serial] =
      q_sim_batch_->CalcDynamicsSerial(x_batch_, u_batch_, sim_params_);

  // is_valid.
  CompareIsValid(is_valid_batch_parallel, is_valid_batch_serial);

  // x_next.
  CompareXNext(x_next_batch_parallel, x_next_batch_serial);

  // B.
  CompareMatrices(B_batch_parallel, B_batch_serial, 1e-5);

  // A.
  CompareMatrices(A_batch_parallel, A_batch_serial, 1e-4);
}

/*
 * Compare BatchQuasistaticSimulator::CalcBundledBTrjDirect against
 *        BatchQuasistaticSimulator::CalcBundledBTrjScalarStd.
 * The goal is to ensure that the outcomes of these two functions are the
 * same given the same seed for the random number generator.
 */
TEST_P(TestBatchQuasistaticSimulator, TestBundledBTrj) {
  SetUpPlanarHand();

  const int T = 50;
  const int n_samples = 100;
  const int seed = 1;

  const int n_q = q_sim_batch_->get_q_sim().get_plant().num_positions();
  const int n_u = q_sim_batch_->get_q_sim().num_actuated_dofs();
  ASSERT_EQ(n_q, x_batch_.cols());
  ASSERT_EQ(n_u, u_batch_.cols());

  MatrixXd x_trj(T + 1, n_q);
  MatrixXd u_trj(T, n_u);

  x_trj.rowwise() = x_batch_.row(0);
  u_trj.rowwise() = u_batch_.row(0);

  sim_params_.gradient_mode = GradientMode::kBOnly;
  auto [A_bundled1, B_bundled1, c_bundled1] =
      q_sim_batch_->CalcBundledABcTrjScalarStd(x_trj.topRows(T), u_trj, 0.1,
                                               sim_params_, n_samples, seed);
  auto B_bundled2 = q_sim_batch_->CalcBundledBTrjDirect(
      x_trj.topRows(T), u_trj, 0.1, sim_params_, n_samples, seed);

  const double tol = sim_params_.use_free_solvers? 1e-8 : 1e-10;
  for (int i = 0; i < T; i++) {
    double err = (B_bundled1[i] - B_bundled2[i]).norm();
    EXPECT_LT(err, tol);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BatchQuasistaticSimulator, TestBatchQuasistaticSimulator,
    testing::ValuesIn(qsim::test::get_use_free_solvers_values()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
