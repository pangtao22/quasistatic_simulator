#include <vector>

#include "get_model_paths.h"
#include "quasistatic_parser.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::cout;
using std::endl;

int main() {
  auto q_model_path =
      GetQsimModelsPath() / "q_sys" / "allegro_hand_tilted_and_sphere.yml";
  cout << q_model_path << endl;
  auto parser = QuasistaticParser(q_model_path);
  auto q_sim = parser.MakeSimulator();

  auto &sim_params = q_sim->get_mutable_sim_params();
  sim_params.h = 0.1;
  sim_params.log_barrier_weight = 116.157;
  sim_params.forward_mode = ForwardDynamicsMode::kLogPyramidMy;
  sim_params.gradient_mode = GradientMode::kAB;

  const auto n_q = q_sim->get_plant().num_positions();
  const auto n_a = q_sim->num_actuated_dofs();
  VectorXd q0(n_q), u0(n_a);
  q0 << -0.07439061,  0.60355771,  0.719262,  0.83094604,  0.48799869,
      1.04966671,  0.62751413,  0.88485928, -0.1868239 ,  0.56199777,
      0.6213565 ,  0.77378053, -0.05885874,  0.67842715,  0.85774352,
      0.98863791,  0.98753373, -0.05247661,  0.13827894,  0.05387271,
      -0.10219377,  0.01038058,  0.06025127;

  u0 <<-0.11350021,  0.63456847,  0.53900271,  0.74822424,  0.61714401,
      0.82656415,  0.54608412,  0.72634116, -0.28723053,  0.78610893,
      0.70629836,  0.65223152,  0.01113549,  0.91425955,  0.90519647,
      0.80027765;

  const auto q_a_cmd_dict = q_sim->GetQaCmdDictFromVec(u0);

  q_sim->CalcDynamics(q0, u0, sim_params);
  cout << "A_max " << q_sim->get_Dq_nextDq().array().abs().maxCoeff();
  cout << "A\n" << q_sim->get_Dq_nextDq() << "\n";
  cout << "B\n" << q_sim->get_Dq_nextDqa_cmd() << endl;

  return 0;
}