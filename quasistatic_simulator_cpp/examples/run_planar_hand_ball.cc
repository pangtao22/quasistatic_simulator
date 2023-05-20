#include <chrono>
#include <filesystem>

#include "qsim/get_model_paths.h"
#include "qsim/quasistatic_simulator.h"

using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;

static const char* kObjectSdfPath =
    (GetQsimModelsPath() / "sphere_yz_rotation_r_0.25m.sdf").c_str();

static const char* kModelDirectivePath =
    (GetQsimModelsPath() / "planar_hand.yml").c_str();

int main() {
  QuasistaticSimParameters sim_params;
  sim_params.h = 0.1;
  sim_params.gravity = Vector3d(0, 0, -10);
  sim_params.nd_per_contact = 2;
  sim_params.contact_detection_tolerance = 1.0;
  sim_params.is_quasi_dynamic = true;

  VectorXd Kp;
  Kp.resize(2);
  Kp << 50, 25;
  const string robot_l_name = "arm_left";
  const string robot_r_name = "arm_right";

  std::unordered_map<string, VectorXd> robot_stiffness_dict = {
      {robot_l_name, Kp}, {robot_r_name, Kp}};

  const string object_name("sphere");
  std::unordered_map<string, string> object_sdf_dict;
  object_sdf_dict[object_name] = kObjectSdfPath;

  auto q_sim = QuasistaticSimulator(kModelDirectivePath, robot_stiffness_dict,
                                    object_sdf_dict, sim_params);

  const auto name_to_idx_map = q_sim.GetModelInstanceNameToIndexMap();
  const auto idx_l = name_to_idx_map.at(robot_l_name);
  const auto idx_r = name_to_idx_map.at(robot_r_name);
  const auto idx_o = name_to_idx_map.at(object_name);

  ModelInstanceIndexToVecMap q0_dict = {{idx_o, Vector3d(0, 0.35, 0)},
                                        {idx_l, Vector2d(-M_PI / 4, -M_PI / 4)},
                                        {idx_r, Vector2d(M_PI / 4, M_PI / 4)}};

  for (int gradient_mode = 0; gradient_mode < 3; gradient_mode++) {
    sim_params.gradient_mode = GradientMode(gradient_mode);
    q_sim.update_sim_params(sim_params);
    cout << "--------------- gradient mode " << gradient_mode << " ---------\n";
    auto t_start = std::chrono::high_resolution_clock::now();
    const int n = 100;
    for (int i = 0; i < n; i++) {
      q_sim.UpdateMbpPositions(q0_dict);
      ModelInstanceIndexToVecMap tau_ext_dict = q_sim.CalcTauExt({});
      q_sim.Step(q0_dict, tau_ext_dict);
      auto q_next_dict = q_sim.GetMbpPositions();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    cout << "wall time microseconds per dynamics: "
         << std::chrono::duration_cast<std::chrono::microseconds>(t_end -
                                                                  t_start)
                    .count() /
                n
         << endl;

    cout << "Dq_nextDq\n" << q_sim.get_Dq_nextDq() << endl;
    cout << "Dq_nextDqa_cmd\n" << q_sim.get_Dq_nextDqa_cmd() << endl;
  }

  return 0;
}
