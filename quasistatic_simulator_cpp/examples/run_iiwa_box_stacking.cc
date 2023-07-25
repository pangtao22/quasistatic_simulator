#include <filesystem>

#include "qsim/get_model_paths.h"
#include "qsim/quasistatic_parser.h"

using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;
using std::filesystem::path;

ModelInstanceIndexToVecMap CreateMapKeyedByModelInstanceIndex(
    const drake::multibody::MultibodyPlant<double>& plant,
    const std::unordered_map<string, VectorXd>& map_str) {
  ModelInstanceIndexToVecMap map_model;
  for (const auto& [name, v] : map_str) {
    auto model = plant.GetModelInstanceByName(name);
    map_model[model] = v;
  }
  return map_model;
}

int main() {
  auto q_model_path = GetQsimModelsPath() / "q_sys" / "iiwa_and_boxes.yml";

  QuasistaticSimParameters sim_params;
  sim_params.h = 0.1;
  sim_params.gravity = Vector3d(0, 0, -10);
  sim_params.nd_per_contact = 4;
  sim_params.contact_detection_tolerance = 0.02;
  sim_params.is_quasi_dynamic = false;

  auto q_parser = QuasistaticParser(q_model_path);
  q_parser.set_sim_params(sim_params);

  std::unique_ptr<QuasistaticSimulator> q_sim_ptr = q_parser.MakeSimulator();
  auto& q_sim = *q_sim_ptr;

  const string iiwa_name("iiwa");
  const string schunk_name("schunk");

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
