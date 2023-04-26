#include "get_model_paths.h"
#include "quasistatic_parser.h"
#include "quasistatic_simulator.h"

#include <drake/geometry/shape_specification.h>

using Eigen::Vector2d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::cout;
using std::endl;

int main() {
  auto q_model_path = GetQsimModelsPath() / "q_sys" / "two_spheres_yz.yml";
  auto parser = QuasistaticParser(q_model_path);
  auto q_sim = parser.MakeSimulator();

  auto &sim_params = q_sim->get_mutable_sim_params();
  sim_params.h = 0.1;
  sim_params.forward_mode = ForwardDynamicsMode::kQpMp;
  sim_params.gradient_mode = GradientMode::kAB;

  const auto n_q = q_sim->get_plant().num_positions();
  const auto n_a = q_sim->num_actuated_dofs();

  std::string robot_name = "sphere_yz_actuated";
  std::string object_name = "sphere_yz";

  auto& plant = q_sim->get_plant();
  auto idx_a = plant.GetModelInstanceByName(robot_name);
  auto idx_u = plant.GetModelInstanceByName(object_name);

  Vector2d q_a0(0, 0);
  Vector2d q_u0(0.3, 0);

  ModelInstanceIndexToVecMap q0_dict = {{idx_a, q_a0}, {idx_u, q_u0}};
  auto q0 = q_sim->GetQVecFromDict(q0_dict);
  Vector2d u0(0.1, 0);

  auto& sg = q_sim->get_mutable_scene_graph();
  q_sim->UpdateMbpPositions(q0_dict);
//
  auto& query_object = q_sim->get_query_object();
  auto sdps = query_object.ComputeSignedDistancePairwiseClosestPoints(
      std::numeric_limits<double>::infinity()
      );
  cout << "num collision pairs " << sdps.size() << endl;
  auto& sdp = sdps[0];
  cout << "ok here's the distance " << sdp.distance << endl;

  // Okay let's try domain randomization.
  auto& diagram = q_sim->get_mutable_diagram();
  auto context = diagram.CreateDefaultContext();
  auto& context_sg = sg.GetMyMutableContextFromRoot(context.get());
  auto& context_plant = plant.GetMyMutableContextFromRoot(context.get());
  sg.ChangeShape(
      &context_sg,
      plant.get_source_id().value(),
      sdp.id_B,
      drake::geometry::Sphere(0.11)
      );

  plant.SetPositions(&context_plant, q0);
  auto& query_object_new = sg.get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
      context_sg);
  auto sdps_new = query_object_new.ComputeSignedDistancePairwiseClosestPoints(100);
  sdp = sdps_new[0];
  cout << "what about the distance now? " << sdp.distance << endl;

//
  auto q_next = q_sim->CalcDynamics(q0, u0, sim_params);
//  auto A = q_sim->get_Dq_nextDq();
//  auto B = q_sim->get_Dq_nextDqa_cmd();
//
//  cout << "A\n" << A << endl;
//  cout << "B\n" << B << endl;

  return 0;
}
