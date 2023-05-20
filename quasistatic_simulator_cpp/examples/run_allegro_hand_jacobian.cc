#include <vector>

#include "qsim/get_model_paths.h"
#include "qsim/quasistatic_parser.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

int main() {
  auto q_model_path =
      GetQsimModelsPath() / "q_sys" / "allegro_hand_and_sphere.yml";
  cout << q_model_path << endl;
  auto parser = QuasistaticParser(q_model_path);
  auto q_sim = parser.MakeSimulator();

  auto& sim_params = q_sim->get_mutable_sim_params();
  sim_params.h = 0.1;
  sim_params.log_barrier_weight = 100;
  sim_params.forward_mode = ForwardDynamicsMode::kQpMp;
  sim_params.gradient_mode = GradientMode::kAB;

  const auto n_q = q_sim->get_plant().num_positions();
  const auto n_a = q_sim->num_actuated_dofs();
  VectorXd q0(n_q), u0(n_a);
  q0 << 3.50150400e-02, 7.52765650e-01, 7.41462320e-01, 8.32610020e-01,
      6.32562690e-01, 1.02378254e+00, 6.40895550e-01, 8.24447820e-01,
      -1.43872500e-01, 7.46968120e-01, 6.19088270e-01, 7.00642790e-01,
      -6.92254100e-02, 7.85331420e-01, 8.29428630e-01, 9.04154360e-01,
      0.98334744, 0.0342708, 0.10602051, 0.14357218,  //  <- quaternion.
      -9.00000000e-02, 1.00000000e-03, 8.00000000e-02;

  u0 << 0.03501504, 0.75276565, 0.74146232, 0.83261002, 0.63256269, 1.02378254,
      0.64089555, 0.82444782, -0.1438725, 0.74696812, 0.61908827, 0.70064279,
      -0.06922541, 0.78533142, 0.82942863, 0.90415436;

  const auto q_a_cmd_dict = q_sim->GetQaCmdDictFromVec(u0);

  // Jacobian investigation.
  const auto& plant = q_sim->get_plant();
  const auto context = plant.CreateDefaultContext();
  const auto name_to_idx_map = q_sim->GetModelInstanceNameToIndexMap();
  const auto model_o = name_to_idx_map.at("sphere");
  auto ball_body_idx = plant.GetBodyIndices(model_o)[0];
  plant.SetPositions(context.get(), q0);
  MatrixXd Jv(6, plant.num_velocities());
  MatrixXd Jq(6, plant.num_positions());
  plant.CalcJacobianSpatialVelocity(
      *context, drake::multibody::JacobianWrtVariable::kV,
      plant.get_body(ball_body_idx).body_frame(), Eigen::Vector3d::Zero(),
      plant.world_frame(), plant.world_frame(), &Jv);
  plant.CalcJacobianSpatialVelocity(
      *context, drake::multibody::JacobianWrtVariable::kQDot,
      plant.get_body(ball_body_idx).body_frame(), Eigen::Vector3d::Zero(),
      plant.world_frame(), plant.world_frame(), &Jq);

  cout << "Jv\n" << Jv << "\n";
  cout << "Jq\n" << Jq << endl;

  cout << "Jq_my\n" << q_sim->ConvertColVToQdot(q_sim->GetQDictFromVec(q0), Jv);

  q_sim->CalcDynamics(q0, u0, sim_params);
  cout << "A\n" << q_sim->get_Dq_nextDq() << "\n";
  cout << "B\n" << q_sim->get_Dq_nextDqa_cmd() << endl;

  return 0;
}
