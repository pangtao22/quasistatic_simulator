#include <chrono>

#include "batch_quasistatic_simulator.h"
#include "get_model_paths.h"
#include "quasistatic_simulator.h"

using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;

const string kObjectSdfPath = GetQsimModelsPath() / "box_1m.sdf";

const string kModelDirectivePath =
    GetQsimModelsPath() / "three_link_arm_and_ground.yml";

int main() {
  QuasistaticSimParameters sim_params;
  sim_params.h = 0.1;
  sim_params.gravity = Vector3d(0, 0, -10);
  sim_params.nd_per_contact = 4;
  sim_params.contact_detection_tolerance = INFINITY;
  sim_params.is_quasi_dynamic = true;
  sim_params.gradient_mode = GradientMode::kBOnly;

  VectorXd Kp;
  Kp.resize(3);
  Kp << 1000, 1000, 1000;
  const string robot_name("arm");

  std::unordered_map<string, VectorXd> robot_stiffness_dict = {
      {robot_name, Kp}};

  const string object_name("box0");
  std::unordered_map<string, string> object_sdf_dict;
  object_sdf_dict[object_name] = kObjectSdfPath;

  auto q_sim = QuasistaticSimulator(kModelDirectivePath, robot_stiffness_dict,
                                    object_sdf_dict, sim_params);

  const auto name_to_idx_map = q_sim.GetModelInstanceNameToIndexMap();
  const auto idx_r = name_to_idx_map.at(robot_name);
  const auto idx_o = name_to_idx_map.at(object_name);

  VectorXd q_u0(7);
  q_u0 << 1, 0, 0, 0, 0.0, 1.7, 0.5;
  ModelInstanceIndexToVecMap q0_dict = {
      {idx_o, q_u0},
      {idx_r, Vector3d(M_PI / 2, -M_PI / 2, -M_PI / 2)},
  };

  const int n_tasks = 20;
  const int n_samples = 100;

  auto q_sim_batch = BatchQuasistaticSimulator(
      kModelDirectivePath, robot_stiffness_dict, object_sdf_dict, sim_params);

  MatrixXd x_batch(n_tasks, 10);
  MatrixXd u_batch(n_tasks, 3);
  for (int i = 0; i < n_tasks; i++) {
    x_batch.row(i).head(3) = q0_dict[idx_r];
    x_batch.row(i).tail(7) = q0_dict[idx_o];
    u_batch.row(i) = q0_dict[idx_r];
  }

  auto t_start = std::chrono::steady_clock::now();
  auto result1 = q_sim_batch.CalcBundledABcTrjScalarStd(
      x_batch, u_batch, 0.1, sim_params, n_samples, 1);
  auto t_end = std::chrono::steady_clock::now();
  cout << "CalcBundledBTrjScalarStd wall time ms: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
              .count()
       << endl;
  const auto &B_batch1 = std::get<1>(result1);

  t_start = std::chrono::steady_clock::now();
  auto B_batch2 = q_sim_batch.CalcBundledBTrjDirect(x_batch, u_batch, 0.1,
                                                    sim_params, n_samples, 1);
  t_end = std::chrono::steady_clock::now();
  cout << "CalcBundledBTrjDirect wall time ms: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
              .count()
       << endl;

  // Diff
  cout << "B errors" << endl;
  for (int i = 0; i < n_tasks; i++) {
    cout << (B_batch1[i] - B_batch2[i]).norm() << endl;
  }

  //  cout << "==Time single-thread execution==" << endl;
  //  auto t_start = std::chrono::steady_clock::now();
  //  auto result1 = q_sim_batch.CalcDynamicsSerial(x_batch, u_batch, 0.1,
  //                                                      GradientMode::kBOnly);
  //  auto t_end = std::chrono::steady_clock::now();
  //  cout << "wall time ms serial: "
  //       << std::chrono::duration_cast<std::chrono::milliseconds>(t_end -
  //           t_start)
  //           .count()
  //       << endl;
  //
  //
  //  cout << "==Time batch execution==" << endl;
  //  t_start = std::chrono::steady_clock::now();
  //  auto result2 = q_sim_batch.CalcDynamicsParallel(x_batch, u_batch, 0.1,
  //                                                 GradientMode::kBOnly);
  //  t_end = std::chrono::steady_clock::now();
  //  cout << "wall time ms parallel: "
  //       << std::chrono::duration_cast<std::chrono::milliseconds>(t_end -
  //           t_start)
  //           .count()
  //       << endl;

  return 0;
}