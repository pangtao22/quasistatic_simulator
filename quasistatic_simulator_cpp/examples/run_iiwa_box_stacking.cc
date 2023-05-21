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
using std::filesystem::path;

static const char* kBox6cmPath =
    (GetQsimModelsPath() / path("box_0.06m.sdf")).c_str();
static const char* kBox7cmPath =
    (GetQsimModelsPath() / path("box_0.07m.sdf")).c_str();
static const char* kBox8cmPath =
    (GetQsimModelsPath() / path("box_0.08m.sdf")).c_str();
static const char* kModelDirectivePath =
    (GetQsimModelsPath() / path("iiwa_and_schunk_and_ground.yml")).c_str();

std::unordered_map<ModelInstanceIndex, VectorXd>
CreateMapKeyedByModelInstanceIndex(
    const drake::multibody::MultibodyPlant<double>& plant,
    const std::unordered_map<string, VectorXd>& map_str) {
  std::unordered_map<ModelInstanceIndex, VectorXd> map_model;
  for (const auto& [name, v] : map_str) {
    auto model = plant.GetModelInstanceByName(name);
    map_model[model] = v;
  }
  return map_model;
}

int main() {
  cout << std::filesystem::current_path().generic_string() << endl;

  QuasistaticSimParameters sim_params;
  sim_params.h = 0.1;
  sim_params.gravity = Vector3d(0, 0, -10);
  sim_params.nd_per_contact = 4;
  sim_params.contact_detection_tolerance = 0.02;
  sim_params.is_quasi_dynamic = false;

  const string iiwa_name("iiwa");
  const string schunk_name("schunk");
  VectorXd Kp_iiwa(7);
  Kp_iiwa << 800, 600, 600, 600, 400, 200, 200;
  VectorXd Kp_schunk(2);
  Kp_schunk << 1000, 1000;

  std::unordered_map<string, VectorXd> robot_stiffness_dict = {
      {iiwa_name, Kp_iiwa}, {schunk_name, Kp_schunk}};
  std::unordered_map<string, string> object_sdf_dict = {
      {"box0", kBox6cmPath}, {"box1", kBox8cmPath}, {"box2", kBox7cmPath},
      {"box3", kBox8cmPath}, {"box4", kBox8cmPath}, {"box5", kBox7cmPath},
      {"box6", kBox8cmPath}, {"box7", kBox8cmPath}, {"box8", kBox7cmPath},
      {"box9", kBox8cmPath}};

  auto q_sim = QuasistaticSimulator(kModelDirectivePath, robot_stiffness_dict,
                                    object_sdf_dict, sim_params);

  MatrixXd q_u0_list(10, 7);
  q_u0_list.row(0) << 1, 0, 0, 0, 0.55, 0, 0.03;
  q_u0_list.row(1) << 1, 0, 0, 0, 0.70, 0, 0.04;
  q_u0_list.row(2) << 1, 0, 0, 0, 0.70, 0., 0.115;
  q_u0_list.row(3) << 1, 0, 0, 0, 0.70, 0., 0.19;
  q_u0_list.row(4) << 1, 0, 0, 0, 0.50, -0.2, 0.04;
  q_u0_list.row(5) << 1, 0, 0, 0, 0.50, -0.2, 0.115;
  q_u0_list.row(6) << 1, 0, 0, 0, 0.50, -0.2, 0.19;
  q_u0_list.row(7) << 1, 0, 0, 0, 0.45, 0.2, 0.04;
  q_u0_list.row(8) << 1, 0, 0, 0, 0.45, 0.2, 0.115;
  q_u0_list.row(9) << 1, 0, 0, 0, 0.48, 0.3, 0.04;

  std::unordered_map<string, VectorXd> q0_dict_str;
  for (int i = 0; i < 10; i++) {
    const string name = "box" + std::to_string(i);
    q0_dict_str[name] = q_u0_list.row(i);
  }

  VectorXd q0_iiwa(7);
  q0_iiwa << 0, 0, 0, -1.75, 0, 1.0, 0;
  q0_dict_str[iiwa_name] = q0_iiwa;

  VectorXd q0_schunk(2);
  q0_schunk << -0.04, 0.04;
  q0_dict_str[schunk_name] = q0_schunk;

  auto q0_dict =
      CreateMapKeyedByModelInstanceIndex(q_sim.get_plant(), q0_dict_str);

  q_sim.UpdateMbpPositions(q0_dict);
  ModelInstanceIndexToVecMap tau_ext_dict = q_sim.CalcTauExt({});
  q_sim.Step(q0_dict, tau_ext_dict);

  return 0;
}
