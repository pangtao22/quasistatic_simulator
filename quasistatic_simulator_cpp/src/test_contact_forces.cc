#include <iostream>

#include <gtest/gtest.h>

#include "get_model_paths.h"
#include "quasistatic_parser.h"
#include "quasistatic_simulator.h"
#include "test_utilities.h"

using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;

class TestContactForces : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    const string kQModelPath =
        GetQsimModelsPath() / "q_sys" / "two_spheres_xyz.yml";
    auto parser = QuasistaticParser(kQModelPath);

    sim_params_.h = 0.1;
    sim_params_.gravity = Vector3d(0, 0, -10);
    sim_params_.is_quasi_dynamic = true;
    // So that contact between the small ball and the big ball is ignored.
    sim_params_.contact_detection_tolerance = 0.1;
    sim_params_.use_free_solvers = GetParam();

    const string robot_name("sphere_xyz_actuated");
    const string object_name("sphere_xyz");
    const double r_robot = 0.1; // radius of actuated ball.
    const double r_obj = 0.5;   // radius of un-actuated ball.

    // Initial conditions.
    q_sim_ = parser.MakeSimulator();
    const auto name_to_idx_map = q_sim_->GetModelInstanceNameToIndexMap();
    model_r_ = name_to_idx_map.at(robot_name);
    model_o_ = name_to_idx_map.at(object_name);
    ModelInstanceIndexToVecMap q0_dict = {{model_o_, Vector3d(0, 0, r_obj)},
                                          {model_r_, Vector3d(-2, -2, r_obj)}};

    q0_ = q_sim_->GetQVecFromDict(q0_dict);
    u0_ = q_sim_->GetQaCmdVecFromDict(q0_dict);
  };

  QuasistaticSimParameters sim_params_;
  std::unique_ptr<QuasistaticSimulator> q_sim_;
  ModelInstanceIndex model_r_; // robot model instance index.
  ModelInstanceIndex model_o_; // object model instance index.
  VectorXd q0_, u0_;
};

TEST_P(TestContactForces, TestNormalVsWeight) {
  // Contact force using QP dynamics.
  sim_params_.nd_per_contact = 4;
  sim_params_.forward_mode = ForwardDynamicsMode::kQpMp;
  drake::multibody::ContactResults<double> cr_qp;
  {
    QuasistaticSimulator::CalcDynamics(q_sim_.get(), q0_, u0_, sim_params_);
    cr_qp = q_sim_->get_contact_results();
    ASSERT_EQ(cr_qp.num_point_pair_contacts(), 1);
  }
  const auto &cp_qp = cr_qp.point_pair_contact_info(0);
  const auto &f_B_W_qp = cp_qp.contact_force();

  // Contact force using SOCP dynamics.
  sim_params_.forward_mode = ForwardDynamicsMode::kSocpMp;
  drake::multibody::ContactResults<double> cr_socp;
  {
    QuasistaticSimulator::CalcDynamics(q_sim_.get(), q0_, u0_, sim_params_);
    cr_socp = q_sim_->get_contact_results();
    ASSERT_EQ(cr_socp.num_point_pair_contacts(), 1);
  }
  const auto &cp_socp = cr_socp.point_pair_contact_info(0);
  const auto &f_B_W_socp = cr_socp.point_pair_contact_info(0).contact_force();

  const double force_tol = sim_params_.use_free_solvers ? 1e-4 : 1e-8;
  // QP and SOCP forward dynamics should predict the same contact forces.
  EXPECT_LE((f_B_W_qp - f_B_W_socp).norm(), force_tol);
  EXPECT_EQ(cp_socp.bodyB_index(), cp_qp.bodyB_index());
  EXPECT_EQ(cp_socp.bodyA_index(), cp_qp.bodyA_index());

  // Weight of the object.
  q_sim_->UpdateMbpPositions(q0_);
  auto tau_ext_dict = q_sim_->CalcTauExt({});
  const auto &plant = q_sim_->get_plant();
  auto obj_model_indices = plant.GetBodyIndices(model_o_);

  bool is_B_object =
      std::find(obj_model_indices.begin(), obj_model_indices.end(),
                cp_socp.bodyB_index()) != obj_model_indices.end();

  Vector3d f_Obj_W = f_B_W_socp * (is_B_object ? 1 : -1);

  // The object's contact force should balance its weight.
  EXPECT_LT((f_Obj_W + tau_ext_dict[model_o_]).norm(), force_tol);
}

INSTANTIATE_TEST_SUITE_P(
    ContactForces, TestContactForces,
    testing::ValuesIn(qsim::test::get_use_free_solvers_values()));

// TODO: test the alignment of sliding direction vs force direction in SOCP.

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
