#include "qsim/quasistatic_simulator.h"

#include <set>
#include <vector>

#include "drake/common/drake_path.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/parsing/process_model_directives.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/systems/framework/diagram_builder.h"
#include "qsim/get_model_paths.h"

using drake::AutoDiffXd;
using drake::Matrix3X;
using drake::MatrixX;
using drake::Vector3;
using drake::Vector4;
using drake::VectorX;
using drake::math::ExtractValue;
using drake::math::InitializeAutoDiff;
using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace {
void CreateMbp(drake::systems::DiagramBuilder<double>* builder,
               const string& model_directive_path,
               const std::unordered_map<string, VectorXd>& robot_stiffness_str,
               const std::unordered_map<string, string>& object_sdf_paths,
               const Eigen::Ref<const Vector3d>& gravity,
               drake::multibody::MultibodyPlant<double>** plant,
               drake::geometry::SceneGraph<double>** scene_graph,
               std::set<ModelInstanceIndex>* robot_models,
               std::set<ModelInstanceIndex>* object_models,
               ModelInstanceIndexToVecMap* robot_stiffness) {
  std::tie(*plant, *scene_graph) =
      drake::multibody::AddMultibodyPlantSceneGraph(builder, 1e-3);
  // Set name so that MBP and SceneGraph can be accessed by name.
  (*plant)->set_name(kMultiBodyPlantName);
  (*scene_graph)->set_name(kSceneGraphName);
  auto parser = drake::multibody::Parser(*plant, *scene_graph);
  // TODO(pang): add package paths from yaml file? Hard-coding paths is clearly
  //  not the solution...
  parser.package_map().Add("quasistatic_simulator", GetQsimModelsPath());
  parser.package_map().Add(
      "drake_manipulation_models",
      drake::MaybeGetDrakePath().value() + "/manipulation/models");
  parser.package_map().Add("iiwa_controller", GetRoboticsUtilitiesModelsPath());

  // Objects.
  // Use a Set to sort object names.
  std::set<std::string> object_names;
  for (const auto& item : object_sdf_paths) {
    object_names.insert(item.first);
  }
  for (const auto& name : object_names) {
    object_models->insert(
        parser.AddModelFromFile(object_sdf_paths.at(name), name));
  }

  // Robots.
  drake::multibody::parsing::ProcessModelDirectives(
      drake::multibody::parsing::LoadModelDirectives(model_directive_path),
      *plant, nullptr, &parser);
  for (const auto& [name, Kp] : robot_stiffness_str) {
    auto robot_model = (*plant)->GetModelInstanceByName(name);

    robot_models->insert(robot_model);
    (*robot_stiffness)[robot_model] = Kp;
  }

  // Gravity.
  (*plant)->mutable_gravity_field().set_gravity_vector(gravity);
  (*plant)->Finalize();
}

}  // namespace

std::unique_ptr<QuasistaticSimulator>
QuasistaticSimulator::MakeQuasistaticSimulator(
    const std::string& model_directive_path,
    const std::unordered_map<std::string, Eigen::VectorXd>& robot_stiffness_str,
    const std::unordered_map<std::string, std::string>& object_sdf_paths,
    const QuasistaticSimParameters& sim_params) {
  auto builder = drake::systems::DiagramBuilder<double>();
  drake::multibody::MultibodyPlant<double>* plant_ptr{nullptr};
  drake::geometry::SceneGraph<double>* scene_graph_ptr{nullptr};
  std::set<drake::multibody::ModelInstanceIndex> robot_models;
  std::set<drake::multibody::ModelInstanceIndex> object_models;
  ModelInstanceIndexToVecMap robot_stiffness;

  CreateMbp(&builder, model_directive_path, robot_stiffness_str,
            object_sdf_paths, sim_params.gravity, &plant_ptr, &scene_graph_ptr,
            &robot_models, &object_models, &robot_stiffness);
  std::unique_ptr<drake::systems::Diagram<double>> diagram = builder.Build();

  return std::unique_ptr<QuasistaticSimulator>(new QuasistaticSimulator(
      sim_params, std::move(diagram), plant_ptr, scene_graph_ptr,
      std::move(robot_models), std::move(object_models),
      std::move(robot_stiffness), SolverSelector::MakeSolverSelector()));
}

std::unique_ptr<QuasistaticSimulator> QuasistaticSimulator::Clone() const {
  auto diagram_new = drake::systems::Diagram<double>::Clone(*diagram_);
  auto plant_new_ptr =
      dynamic_cast<const drake::multibody::MultibodyPlant<double>*>(
          &(diagram_new->GetSubsystemByName(plant_->get_name())));
  auto sg_new_ptr = dynamic_cast<const drake::geometry::SceneGraph<double>*>(
      &(diagram_new->GetSubsystemByName(sg_->get_name())));

  return std::unique_ptr<QuasistaticSimulator>(new QuasistaticSimulator(
      sim_params_, std::move(diagram_new), plant_new_ptr, sg_new_ptr,
      std::move(
          std::set<drake::multibody::ModelInstanceIndex>(models_actuated_)),
      std::move(
          std::set<drake::multibody::ModelInstanceIndex>(models_unactuated_)),
      std::move(ModelInstanceIndexToVecMap(robot_stiffness_)),
      solver_selector_->Clone()));
}

QuasistaticSimulator::QuasistaticSimulator(
    QuasistaticSimParameters sim_params,
    std::unique_ptr<drake::systems::Diagram<double>> diagram,
    const drake::multibody::MultibodyPlant<double>* plant_ptr,
    const drake::geometry::SceneGraph<double>* scene_graph_ptr,
    std::set<drake::multibody::ModelInstanceIndex>&& robot_models,
    std::set<drake::multibody::ModelInstanceIndex>&& object_models,
    ModelInstanceIndexToVecMap&& robot_stiffness,
    std::unique_ptr<SolverSelector> solver_selector)
    : sim_params_(sim_params),
      diagram_(std::move(diagram)),
      plant_(plant_ptr),
      sg_(scene_graph_ptr),
      models_actuated_(robot_models),
      models_unactuated_(object_models),
      robot_stiffness_(robot_stiffness),
      diagram_ad_(
          drake::systems::System<double>::ToAutoDiffXd<drake::systems::Diagram>(
              *diagram_)),
      plant_ad_(
          dynamic_cast<const drake::multibody::MultibodyPlant<AutoDiffXd>*>(
              &(diagram_ad_->GetSubsystemByName(plant_->get_name())))),
      sg_ad_(
          dynamic_cast<const drake::geometry::SceneGraph<drake::AutoDiffXd>*>(
              &(diagram_ad_->GetSubsystemByName(sg_->get_name())))),
      solver_selector_(std::move(solver_selector)),
      solver_log_pyramid_(
          std::make_unique<QpLogBarrierSolver>(*solver_selector_)),
      solver_log_icecream_(
          std::make_unique<SocpLogBarrierSolver>(*solver_selector_)) {
  // Contexts.
  context_ = diagram_->CreateDefaultContext();
  context_plant_ =
      &(diagram_->GetMutableSubsystemContext(*plant_, context_.get()));
  context_sg_ = &(diagram_->GetMutableSubsystemContext(*sg_, context_.get()));

  // AutoDiff contexts.
  context_ad_ = diagram_ad_->CreateDefaultContext();
  context_plant_ad_ =
      &(diagram_ad_->GetMutableSubsystemContext(*plant_ad_, context_ad_.get()));
  context_sg_ad_ =
      &(diagram_ad_->GetMutableSubsystemContext(*sg_ad_, context_ad_.get()));

  // All models instances.
  models_all_ = models_unactuated_;
  models_all_.insert(models_actuated_.begin(), models_actuated_.end());

  // MBP introspection.
  n_q_ = plant_->num_positions();
  n_v_ = plant_->num_velocities();

  for (const auto& model : models_all_) {
    velocity_indices_[model] = GetIndicesForModel(model, ModelIndicesMode::kV);
    position_indices_[model] = GetIndicesForModel(model, ModelIndicesMode::kQ);
  }

  n_v_a_ = 0;
  for (const auto& model : models_actuated_) {
    auto n_v_a_i = plant_->num_velocities(model);
    DRAKE_THROW_UNLESS(n_v_a_i == robot_stiffness_[model].size());
    n_v_a_ += n_v_a_i;
  }

  n_v_u_ = 0;
  for (const auto& model : models_unactuated_) {
    n_v_u_ += plant_->num_velocities(model);
  }

  // Find planar model instances.
  /* Features of a 3D un-actuated model instance:
   *
   * 1. The model instance has only 1 rigid body.
   * 2. The model instance has a floating base.
   * 3. The model instance has 6 velocities and 7 positions.
   */
  for (const auto& model : models_unactuated_) {
    const auto n_v = plant_->num_velocities(model);
    const auto n_q = plant_->num_positions(model);
    if (n_v == 6 && n_q == 7) {
      const auto body_indices = plant_->GetBodyIndices(model);
      DRAKE_THROW_UNLESS(body_indices.size() == 1);
      DRAKE_THROW_UNLESS(plant_->get_body(body_indices.at(0)).is_floating());
      is_3d_floating_[model] = true;
    } else {
      is_3d_floating_[model] = false;
    }
  }

  for (const auto& model : models_actuated_) {
    is_3d_floating_[model] = false;
  }

  // QP derivative.
  dqp_ = std::make_unique<QpDerivativesActive>(
      sim_params_.gradient_lstsq_tolerance);
  dsocp_ =
      std::make_unique<SocpDerivatives>(sim_params_.gradient_lstsq_tolerance);

  // Find smallest stiffness.
  VectorXd min_stiffness_vec(models_actuated_.size());
  int i = 0;
  for (const auto& model : models_actuated_) {
    min_stiffness_vec[i] = robot_stiffness_.at(model).minCoeff();
    i++;
  }
  min_K_a_ = min_stiffness_vec.minCoeff();

  // ContactComputers.
  cjc_ = std::make_unique<ContactJacobianCalculator<double>>(diagram_.get(),
                                                             models_all_);
  cjc_ad_ = std::make_unique<ContactJacobianCalculator<AutoDiffXd>>(
      diagram_ad_.get(), models_all_);

  contact_results_.set_plant(plant_);
}

std::vector<int> QuasistaticSimulator::GetIndicesForModel(
    drake::multibody::ModelInstanceIndex idx, ModelIndicesMode mode) const {
  std::vector<double> selector;
  if (mode == ModelIndicesMode::kQ) {
    selector.resize(n_q_);
  } else {
    selector.resize(n_v_);
  }
  std::iota(selector.begin(), selector.end(), 0);
  Eigen::Map<VectorXd> selector_eigen(selector.data(), selector.size());

  VectorXd indices_d;
  if (mode == ModelIndicesMode::kQ) {
    indices_d = plant_->GetPositionsFromArray(idx, selector_eigen);
  } else {
    indices_d = plant_->GetVelocitiesFromArray(idx, selector_eigen);
  }
  std::vector<int> indices(indices_d.size());
  for (size_t i = 0; i < indices_d.size(); i++) {
    indices[i] = roundl(indices_d[i]);
  }
  return indices;
}

/*
 * Similar to the python implementation, this function updates context_plant_
 * and query_object_.
 */
void QuasistaticSimulator::UpdateMbpPositions(
    const ModelInstanceIndexToVecMap& q_dict) {
  for (const auto& model : models_all_) {
    plant_->SetPositions(context_plant_, model, q_dict.at(model));
  }

  query_object_ =
      &(sg_->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
          *context_sg_));
}

void QuasistaticSimulator::UpdateMbpPositions(
    const Eigen::Ref<const Eigen::VectorXd>& q) {
  plant_->SetPositions(context_plant_, q);
  query_object_ =
      &(sg_->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
          *context_sg_));
}

void QuasistaticSimulator::UpdateMbpAdPositions(
    const ModelInstanceIndexToVecAdMap& q_dict) const {
  for (const auto& model : models_all_) {
    plant_ad_->SetPositions(context_plant_ad_, model, q_dict.at(model));
  }

  query_object_ad_ =
      &(sg_ad_->get_query_output_port()
            .Eval<drake::geometry::QueryObject<AutoDiffXd>>(*context_sg_ad_));
}

void QuasistaticSimulator::UpdateMbpAdPositions(
    const Eigen::Ref<const drake::AutoDiffVecXd>& q) const {
  plant_ad_->SetPositions(context_plant_ad_, q);

  query_object_ad_ =
      &(sg_ad_->get_query_output_port()
            .Eval<drake::geometry::QueryObject<AutoDiffXd>>(*context_sg_ad_));
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetMbpPositions() const {
  ModelInstanceIndexToVecMap q_dict;
  for (const auto& model : models_all_) {
    q_dict[model] = plant_->GetPositions(*context_plant_, model);
  }
  return q_dict;
}

Eigen::VectorXd QuasistaticSimulator::GetPositions(
    drake::multibody::ModelInstanceIndex model) const {
  return plant_->GetPositions(*context_plant_, model);
}

void QuasistaticSimulator::CalcQAndTauH(
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
    const ModelInstanceIndexToVecMap& tau_ext_dict, const double h,
    MatrixXd* Q_ptr, VectorXd* tau_h_ptr,
    const double unactuated_mass_scale) const {
  MatrixXd& Q = *Q_ptr;
  Q = MatrixXd::Zero(n_v_, n_v_);
  VectorXd& tau_h = *tau_h_ptr;
  tau_h = VectorXd::Zero(n_v_);
  ModelInstanceIndexToMatrixMap M_u_dict;
  if (sim_params_.is_quasi_dynamic) {
    M_u_dict = CalcScaledMassMatrix(h, unactuated_mass_scale);
  }

  for (const auto& model : models_unactuated_) {
    const auto& idx_v = velocity_indices_.at(model);
    const auto n_v_i = idx_v.size();
    const VectorXd& tau_ext = tau_ext_dict.at(model);

    for (int i = 0; i < tau_ext.size(); i++) {
      tau_h(idx_v[i]) = tau_ext(i) * h;
    }

    if (sim_params_.is_quasi_dynamic) {
      for (int i = 0; i < n_v_i; i++) {
        for (int j = 0; j < n_v_i; j++) {
          Q(idx_v[i], idx_v[j]) = M_u_dict.at(model)(i, j);
        }
      }
    }
  }

  for (const auto& model : models_actuated_) {
    const auto& idx_v = velocity_indices_.at(model);
    VectorXd dq_a_cmd = q_a_cmd_dict.at(model) - q_dict.at(model);
    const auto& Kp = robot_stiffness_.at(model);
    VectorXd tau_a_h = Kp.array() * dq_a_cmd.array();
    tau_a_h += tau_ext_dict.at(model);
    tau_a_h *= h;

    for (int i = 0; i < tau_a_h.size(); i++) {
      tau_h(idx_v[i]) = tau_a_h(i);
    }

    for (int i = 0; i < idx_v.size(); i++) {
      int idx = idx_v[i];
      Q(idx, idx) = Kp(i) * h * h;
    }
  }
}

void AddPointPairContactInfoFromForce(
    const ContactPairInfo<double>& cpi,
    const Eigen::Ref<const Vector3d>& f_Bc_W,
    drake::multibody::ContactResults<double>* contact_results) {
  drake::geometry::PenetrationAsPointPair<double> papp;
  papp.id_A = cpi.id_A;
  papp.id_B = cpi.id_B;
  papp.p_WCa = cpi.p_WCa;
  papp.p_WCb = cpi.p_WCb;
  papp.nhat_BA_W = cpi.nhat_BA_W;
  Vector3d p_WC = (papp.p_WCa + papp.p_WCb) / 2;
  contact_results->AddContactInfo(
      drake::multibody::PointPairContactInfo<double>(
          cpi.body_A_idx, cpi.body_B_idx, f_Bc_W, p_WC, 0, 0, papp));
}

void QuasistaticSimulator::CalcContactResultsQp(
    const std::vector<ContactPairInfo<double>>& contact_info_list,
    const Eigen::Ref<const Eigen::VectorXd>& beta_star, const int n_d,
    const double h, drake::multibody::ContactResults<double>* contact_results) {
  const auto n_c = contact_info_list.size();
  DRAKE_ASSERT(beta_star.size() == n_c * n_d);
  contact_results->Clear();
  int i_beta = 0;
  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& cpi = contact_info_list[i_c];

    // Compute contact force.
    Vector3d f_Ac_W;
    f_Ac_W.setZero();
    for (int i = 0; i < n_d; i++) {
      f_Ac_W +=
          (cpi.nhat_BA_W + cpi.mu * cpi.t_W.col(i)) * beta_star[i_beta + i];
    }
    f_Ac_W /= h;

    // Assemble Contact info.
    AddPointPairContactInfoFromForce(cpi, -f_Ac_W, contact_results);

    i_beta += n_d;
  }
}

void QuasistaticSimulator::CalcContactResultsSocp(
    const std::vector<ContactPairInfo<double>>& contact_info_list,
    const vector<VectorXd>& lambda_star, const double h,
    drake::multibody::ContactResults<double>* contact_results) {
  const auto n_c = contact_info_list.size();
  DRAKE_ASSERT(n_c == lambda_star.size());
  contact_results->Clear();

  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& cpi = contact_info_list[i_c];

    // Compute contact force.
    Vector3d f_Ac_W = cpi.nhat_BA_W * lambda_star[i_c][0] / cpi.mu;
    f_Ac_W += cpi.t_W * lambda_star[i_c].tail(2);
    f_Ac_W /= h;

    // Assemble Contact info.
    AddPointPairContactInfoFromForce(cpi, -f_Ac_W, contact_results);
  }
}

void QuasistaticSimulator::Step(const ModelInstanceIndexToVecMap& q_a_cmd_dict,
                                const ModelInstanceIndexToVecMap& tau_ext_dict,
                                const QuasistaticSimParameters& params) {
  const auto fm = params.forward_mode;
  const auto q_dict = GetMbpPositions();
  auto q_next_dict(q_dict);

  if (kPyramidModes.find(fm) != kPyramidModes.end()) {
    // Optimization coefficient matrices and vectors.
    MatrixXd Q, Jn, J;
    VectorXd tau_h, phi, phi_constraints;
    // Primal and dual solutions.
    VectorXd v_star;
    CalcPyramidMatrices(q_dict, q_a_cmd_dict, tau_ext_dict, params, &Q, &tau_h,
                        &Jn, &J, &phi, &phi_constraints);

    if (fm == ForwardDynamicsMode::kQpMp) {
      VectorXd beta_star;
      ForwardQp(Q, tau_h, J, phi_constraints, params, &q_next_dict, &v_star,
                &beta_star);

      if (params.calc_contact_forces) {
        CalcContactResultsQp(cjc_->get_contact_pair_info_list(), beta_star,
                             params.nd_per_contact, params.h,
                             &contact_results_);
        contact_results_.set_plant(plant_);
      }

      BackwardQp(Q, tau_h, Jn, J, phi_constraints, q_dict, q_next_dict, v_star,
                 beta_star, params);
      return;
    }

    if (fm == ForwardDynamicsMode::kLogPyramidMp) {
      ForwardLogPyramid(Q, tau_h, J, phi_constraints, params, &q_next_dict,
                        &v_star);
      BackwardLogPyramid(Q, J, phi_constraints, q_dict, q_next_dict, v_star,
                         params, nullptr);
      return;
    }

    if (fm == ForwardDynamicsMode::kLogPyramidMy) {
      ForwardLogPyramidInHouse(Q, tau_h, J, phi_constraints, params,
                               &q_next_dict, &v_star);
      BackwardLogPyramid(Q, J, phi_constraints, q_dict, q_next_dict, v_star,
                         params, &solver_log_pyramid_->get_H_llt());
      return;
    }
  }

  if (kIcecreamModes.find(fm) != kIcecreamModes.end()) {
    MatrixXd Q;
    VectorXd tau_h, phi;
    std::vector<Eigen::Matrix3Xd> J_list;
    VectorXd v_star;
    CalcIcecreamMatrices(q_dict, q_a_cmd_dict, tau_ext_dict, params, &Q, &tau_h,
                         &J_list, &phi);

    if (fm == ForwardDynamicsMode::kSocpMp) {
      std::vector<Eigen::VectorXd> lambda_star_list;
      std::vector<Eigen::VectorXd> e_list;

      ForwardSocp(Q, tau_h, J_list, phi, params, &q_next_dict, &v_star,
                  &lambda_star_list, &e_list);

      if (params.calc_contact_forces) {
        CalcContactResultsSocp(cjc_->get_contact_pair_info_list(),
                               lambda_star_list, params.h, &contact_results_);
        contact_results_.set_plant(plant_);
      }

      BackwardSocp(Q, tau_h, J_list, e_list, phi, q_dict, q_next_dict, v_star,
                   lambda_star_list, params);
      return;
    }

    if (fm == ForwardDynamicsMode::kLogIcecream) {
      ForwardLogIcecream(Q, tau_h, J_list, phi, params, &q_next_dict, &v_star);
      BackwardLogIcecream(q_dict, q_next_dict, v_star, params,
                          solver_log_icecream_->get_H_llt());
      return;
    }
  }

  std::stringstream ss;
  ss << "Forward dynamics mode " << static_cast<int>(fm)
     << " is not supported in C++.";
  throw std::logic_error(ss.str());
}

void QuasistaticSimulator::CalcPyramidMatrices(
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
    const ModelInstanceIndexToVecMap& tau_ext_dict,
    const QuasistaticSimParameters& params, Eigen::MatrixXd* Q,
    Eigen::VectorXd* tau_h_ptr, Eigen::MatrixXd* Jn_ptr, Eigen::MatrixXd* J_ptr,
    Eigen::VectorXd* phi_ptr, Eigen::VectorXd* phi_constraints_ptr) const {
  const auto sdps = CalcCollisionPairs(params.contact_detection_tolerance);
  std::vector<MatrixXd> J_list;
  const auto n_d = params.nd_per_contact;
  cjc_->CalcJacobianAndPhiQp(context_plant_, sdps, n_d, phi_ptr, Jn_ptr,
                             &J_list);
  MatrixXd& J = *J_ptr;
  VectorXd& phi_constraints = *phi_constraints_ptr;

  const auto n_c = J_list.size();
  const auto n_f = n_c * n_d;
  J.resize(n_f, n_v_);
  phi_constraints.resize(n_f);
  for (int i_c = 0; i_c < n_c; i_c++) {
    J(Eigen::seqN(i_c * n_d, n_d), Eigen::all) = J_list[i_c];
    phi_constraints(Eigen::seqN(i_c * n_d, n_d)).setConstant((*phi_ptr)(i_c));
  }

  CalcQAndTauH(q_dict, q_a_cmd_dict, tau_ext_dict, params.h, Q, tau_h_ptr,
               params.unactuated_mass_scale);
}

void QuasistaticSimulator::CalcIcecreamMatrices(
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
    const ModelInstanceIndexToVecMap& tau_ext_dict,
    const QuasistaticSimParameters& params, Eigen::MatrixXd* Q,
    Eigen::VectorXd* tau_h, std::vector<Eigen::Matrix3Xd>* J_list,
    Eigen::VectorXd* phi) const {
  const auto sdps = CalcCollisionPairs(params.contact_detection_tolerance);
  cjc_->CalcJacobianAndPhiSocp(context_plant_, sdps, phi, J_list);
  CalcQAndTauH(q_dict, q_a_cmd_dict, tau_ext_dict, params.h, Q, tau_h,
               params.unactuated_mass_scale);
}

void QuasistaticSimulator::ForwardQp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr,
    Eigen::VectorXd* beta_star_ptr) {
  auto& q_dict = *q_dict_ptr;
  const auto n_f = phi_constraints.size();
  const auto h = params.h;

  // construct and solve MathematicalProgram.
  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");
  prog.AddQuadraticCost(Q, -tau_h, v, true);

  const VectorXd e = phi_constraints / h;
  auto constraints = prog.AddLinearConstraint(
      -J, VectorXd::Constant(n_f, -std::numeric_limits<double>::infinity()), e,
      v);
  const drake::solvers::SolverInterface& solver = PickBestQpSolver(params);
  solver.Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error("Quasistatic dynamics QP cannot be solved.");
  }

  *v_star_ptr = mp_result_.GetSolution(v);
  if (constraints.evaluator()->num_constraints() > 0) {
    *beta_star_ptr = -mp_result_.GetDualSolution(constraints);
  } else {
    *beta_star_ptr = Eigen::VectorXd(0);
  }

  // Update q_dict.
  UpdateQdictFromV(*v_star_ptr, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::ForwardSocp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const std::vector<Eigen::Matrix3Xd>& J_list,
    const Eigen::Ref<const Eigen::VectorXd>& phi,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr,
    std::vector<Eigen::VectorXd>* lambda_star_ptr,
    std::vector<Eigen::VectorXd>* e_list) {
  auto& q_dict = *q_dict_ptr;
  VectorXd& v_star = *v_star_ptr;
  const auto h = params.h;
  const auto n_c = phi.size();

  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");

  prog.AddQuadraticCost(Q, -tau_h, v, true);

  std::vector<drake::solvers::Binding<drake::solvers::LorentzConeConstraint>>
      constraints;
  for (int i_c = 0; i_c < n_c; i_c++) {
    const double mu = cjc_->get_friction_coefficient(i_c);
    e_list->emplace_back(Vector3d(phi[i_c] / mu / h, 0, 0));
    constraints.push_back(
        prog.AddLorentzConeConstraint(J_list.at(i_c), e_list->back(), v));
  }

  const drake::solvers::SolverInterface& solver = PickBestSocpSolver(params);
  solver.Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error("Quasistatic dynamics SOCP cannot be solved.");
  }

  // Primal and dual solutions.
  v_star = mp_result_.GetSolution(v);
  if (is_socp_calculating_dual(params)) {
    for (int i = 0; i < n_c; i++) {
      lambda_star_ptr->emplace_back(mp_result_.GetDualSolution(constraints[i]));
    }
  } else {
    lambda_star_ptr->clear();
  }

  // Update q_dict.
  UpdateQdictFromV(v_star, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::ForwardLogPyramid(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr) {
  auto& q_dict = *q_dict_ptr;
  VectorXd& v_star = *v_star_ptr;
  const auto n_f = J.rows();
  const auto h = params.h;

  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");
  auto s = prog.NewContinuousVariables(n_f, "s");

  prog.AddQuadraticCost(Q, -tau_h, v, true);
  prog.AddLinearCost(-VectorXd::Constant(n_f, 1 / params.log_barrier_weight), 0,
                     s);

  drake::solvers::VectorXDecisionVariable v_s_i(n_v_ + 1);
  v_s_i.head(n_v_) = v;
  for (int i = 0; i < n_f; i++) {
    MatrixXd A = MatrixXd::Zero(3, n_v_ + 1);
    A.row(0).head(n_v_) = J.row(i);
    A(2, n_v_) = 1;

    Vector3d b(phi_constraints[i] / h, 1, 0);

    v_s_i[n_v_] = s[i];
    prog.AddExponentialConeConstraint(A.sparseView(), b, v_s_i);
  }
  const drake::solvers::SolverInterface& solver = PickBestConeSolver(params);
  solver.Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error(
        "Quasistatic dynamics Log Pyramid cannot be solved.");
  }

  v_star = mp_result_.GetSolution(v);

  // Update q_dict.
  UpdateQdictFromV(v_star, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::ForwardLogPyramidInHouse(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr) {
  auto& q_dict = *q_dict_ptr;

  solver_log_pyramid_->Solve(Q, -tau_h, -J, phi_constraints / params.h,
                             params.log_barrier_weight, params.use_free_solvers,
                             v_star_ptr);

  // Update q_dict.
  UpdateQdictFromV(*v_star_ptr, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::ForwardLogIcecream(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const std::vector<Eigen::Matrix3Xd>& J_list,
    const Eigen::Ref<const Eigen::VectorXd>& phi,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr) {
  auto& q_dict = *q_dict_ptr;

  const auto h = params.h;
  const auto n_c = J_list.size();
  const auto n_v = Q.rows();

  MatrixXd J(n_c * 3, n_v);
  VectorXd phi_h_mu(n_c);
  for (int i = 0; i < n_c; i++) {
    J.block(i * 3, 0, 3, n_v) = J_list.at(i);
    phi_h_mu[i] = phi[i] / h / cjc_->get_friction_coefficient(i);
  }

  solver_log_icecream_->Solve(Q, -tau_h, -J, phi_h_mu,
                              params.log_barrier_weight,
                              params.use_free_solvers, v_star_ptr);

  // Update q_dict.
  UpdateQdictFromV(*v_star_ptr, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::BackwardQp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const Eigen::Ref<const Eigen::MatrixXd>& Jn,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_dict_next,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const Eigen::Ref<const Eigen::VectorXd>& lambda_star,
    const QuasistaticSimParameters& params) {
  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }
  const auto h = params.h;
  const auto n_d = params.nd_per_contact;

  if (params.gradient_mode == GradientMode::kAB) {
    dqp_->UpdateProblem(Q, -tau_h, -J, phi_constraints / h, v_star, lambda_star,
                        0.1 * params.h, true);
    const auto& Dv_nextDe = dqp_->get_DzDe();
    const auto& Dv_nextDb = dqp_->get_DzDb();

    Dq_nextDq_ =
        CalcDfDxQp(Dv_nextDb, Dv_nextDe, Jn, v_star, q_dict_next, h, n_d);
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, h, q_dict_next);
    return;
  }

  if (params.gradient_mode == GradientMode::kBOnly) {
    dqp_->UpdateProblem(Q, -tau_h, -J, phi_constraints / h, v_star, lambda_star,
                        0.1 * params.h, false);
    const auto& Dv_nextDb = dqp_->get_DzDb();
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, h, q_dict_next);
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    return;
  }

  throw std::runtime_error("Invalid gradient_mode.");
}

void QuasistaticSimulator::BackwardSocp(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
    const vector<Eigen::Matrix3Xd>& J_list,
    const std::vector<Eigen::VectorXd>& e_list,
    const Eigen::Ref<const Eigen::VectorXd>& phi,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_dict_next,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const std::vector<Eigen::VectorXd>& lambda_star_list,
    const QuasistaticSimParameters& params) {
  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }

  std::vector<Eigen::MatrixXd> G_list;
  for (const auto& J : J_list) {
    G_list.emplace_back(-J);
  }

  if (params.gradient_mode == GradientMode::kBOnly) {
    dsocp_->UpdateProblem(Q, -tau_h, G_list, e_list, v_star, lambda_star_list,
                          0.1 * params.h, false);
    const auto& Dv_nextDb = dsocp_->get_DzDb();
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, params.h, q_dict_next);
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    return;
  }

  if (params.gradient_mode == GradientMode::kAB) {
    dsocp_->UpdateProblem(Q, -tau_h, G_list, e_list, v_star, lambda_star_list,
                          0.1 * params.h, true);
    const auto& Dv_nextDb = dsocp_->get_DzDb();
    const auto& Dv_nextDe = dsocp_->get_DzDe();
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, params.h, q_dict_next);
    Dq_nextDq_ = CalcDfDxSocp(Dv_nextDb, Dv_nextDe, J_list, v_star, q_dict,
                              q_dict_next, params.h);

    return;
  }

  throw std::runtime_error("Invalid gradient_mode.");
}

void QuasistaticSimulator::BackwardLogPyramid(
    const Eigen::Ref<const Eigen::MatrixXd>& Q,
    const Eigen::Ref<const Eigen::MatrixXd>& J,
    const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const QuasistaticSimParameters& params,
    Eigen::LLT<Eigen::MatrixXd> const* const H_llt) {
  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }

  if (H_llt) {
    CalcUnconstrainedBFromHessian(*H_llt, params, q_dict, &Dq_nextDqa_cmd_);
    if (params.gradient_mode == GradientMode::kAB) {
      Dq_nextDq_ =
          CalcDfDxLogPyramid(v_star, q_dict, q_next_dict, params, *H_llt);
    } else {
      Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    }
    return;
  }

  Eigen::MatrixXd H(n_v_, n_v_);
  // not used, but needed by CalcGradientAndHessian.
  Eigen::VectorXd Df(n_v_);
  solver_log_pyramid_->CalcGradientAndHessian(
      Q, VectorXd::Zero(n_v_), -J, phi_constraints / params.h, v_star,
      params.log_barrier_weight, &Df, &H);

  CalcUnconstrainedBFromHessian(H.llt(), params, q_dict, &Dq_nextDqa_cmd_);
  if (params.gradient_mode == GradientMode::kAB) {
    Dq_nextDq_ =
        CalcDfDxLogPyramid(v_star, q_dict, q_next_dict, params, H.llt());
  } else {
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
  }
}

void QuasistaticSimulator::BackwardLogIcecream(
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const QuasistaticSimParameters& params,
    const Eigen::LLT<Eigen::MatrixXd>& H_llt) {
  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }

  if (params.gradient_mode == GradientMode::kBOnly) {
    CalcUnconstrainedBFromHessian(H_llt, params, q_next_dict, &Dq_nextDqa_cmd_);
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    return;
  }

  if (params.gradient_mode == GradientMode::kAB) {
    CalcUnconstrainedBFromHessian(H_llt, params, q_dict, &Dq_nextDqa_cmd_);
    Dq_nextDq_ = CalcDfDxLogIcecream(v_star, q_dict, q_next_dict, params.h,
                                     params.log_barrier_weight, H_llt);
    return;
  }

  throw std::logic_error("Invalid gradient_mode.");
}

void QuasistaticSimulator::CalcUnconstrainedBFromHessian(
    const Eigen::LLT<Eigen::MatrixXd>& H_llt,
    const QuasistaticSimParameters& params,
    const ModelInstanceIndexToVecMap& q_dict, Eigen::MatrixXd* B_ptr) const {
  MatrixXd Dv_nextDb(n_v_, n_v_);
  Dv_nextDb.setIdentity();
  Dv_nextDb *= -params.log_barrier_weight;
  H_llt.solveInPlace(Dv_nextDb);
  *B_ptr = CalcDfDu(Dv_nextDb, params.h, q_dict);
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetVdictFromVec(
    const Eigen::Ref<const Eigen::VectorXd>& v) const {
  DRAKE_THROW_UNLESS(v.size() == n_v_);
  ModelInstanceIndexToVecMap v_dict;

  for (const auto& model : models_all_) {
    const auto& idx_v = velocity_indices_.at(model);
    v_dict[model] = v(idx_v);
  }
  return v_dict;
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetQDictFromVec(
    const Eigen::Ref<const Eigen::VectorXd>& q) const {
  DRAKE_THROW_UNLESS(q.size() == n_q_);
  ModelInstanceIndexToVecMap q_dict;

  for (const auto& model : models_all_) {
    const auto& idx_q = position_indices_.at(model);
    q_dict[model] = q(idx_q);
  }
  return q_dict;
}

Eigen::VectorXd QuasistaticSimulator::GetQVecFromDict(
    const ModelInstanceIndexToVecMap& q_dict) const {
  VectorXd q(n_q_);
  for (const auto& model : models_all_) {
    q(position_indices_.at(model)) = q_dict.at(model);
  }
  return q;
}

Eigen::VectorXd QuasistaticSimulator::GetQaCmdVecFromDict(
    const ModelInstanceIndexToVecMap& q_a_cmd_dict) const {
  int i_start = 0;
  VectorXd q_a_cmd(n_v_a_);
  for (const auto& model : models_actuated_) {
    auto n_v_i = plant_->num_velocities(model);
    q_a_cmd.segment(i_start, n_v_i) = q_a_cmd_dict.at(model);
    i_start += n_v_i;
  }

  return q_a_cmd;
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetQaCmdDictFromVec(
    const Eigen::Ref<const Eigen::VectorXd>& q_a_cmd) const {
  ModelInstanceIndexToVecMap q_a_cmd_dict;
  int i_start = 0;
  for (const auto& model : models_actuated_) {
    auto n_v_i = plant_->num_velocities(model);
    q_a_cmd_dict[model] = q_a_cmd.segment(i_start, n_v_i);
    i_start += n_v_i;
  }

  return q_a_cmd_dict;
}

Eigen::VectorXi QuasistaticSimulator::GetModelsIndicesIntoQ(
    const std::set<drake::multibody::ModelInstanceIndex>& models) const {
  const int n = std::accumulate(models.begin(), models.end(), 0,
                                [&position_indices = position_indices_](
                                    int n, const ModelInstanceIndex& model) {
                                  return n + position_indices.at(model).size();
                                });
  Eigen::VectorXi models_indices(n);
  Eigen::Index i_start = 0;
  for (const auto& model : models) {
    const auto& indices = position_indices_.at(model);
    const auto n_model = indices.size();
    models_indices(Eigen::seqN(i_start, n_model)) =
        Eigen::Map<const Eigen::VectorXi>(indices.data(), n_model);
    i_start += n_model;
  }
  return models_indices;
}

Eigen::VectorXi QuasistaticSimulator::GetQaIndicesIntoQ() const {
  return GetModelsIndicesIntoQ(models_actuated_);
}

Eigen::VectorXi QuasistaticSimulator::GetQuIndicesIntoQ() const {
  return GetModelsIndicesIntoQ(models_unactuated_);
}

void QuasistaticSimulator::Step(
    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
    const ModelInstanceIndexToVecMap& tau_ext_dict) {
  Step(q_a_cmd_dict, tau_ext_dict, sim_params_);
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDu(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb, const double h,
    const ModelInstanceIndexToVecMap& q_dict) const {
  MatrixXd DbDqa_cmd = MatrixXd::Zero(n_v_, n_v_a_);
  int j_start = 0;
  for (const auto& model : models_actuated_) {
    const auto& idx_v = velocity_indices_.at(model);
    const int n_v_i = idx_v.size();
    const auto& Kq_i = robot_stiffness_.at(model);

    for (int k = 0; k < n_v_i; k++) {
      int i = idx_v[k];
      int j = j_start + k;
      DbDqa_cmd(i, j) = -h * Kq_i[k];
    }

    j_start += n_v_i;
  }

  const MatrixXd Dv_nextDqa_cmd = Dv_nextDb * DbDqa_cmd;

  // 2D systems.
  if (n_v_ == n_q_) {
    return h * Dv_nextDqa_cmd;
  }

  // 3D systems.
  return h * ConvertRowVToQdot(q_dict, Dv_nextDqa_cmd);
}

/*
 * Used to provide the optional input to CalcDGactiveDqFromJActiveList, when
 * the dynamics is a QP.
 */
std::vector<std::vector<int>> CalcRelativeActiveIndicesList(
    const std::vector<int>& lambda_star_active_indices, const int n_d) {
  int i_c_current = -1;
  std::vector<std::vector<int>> relative_active_indices_list;
  for (const auto i : lambda_star_active_indices) {
    const int i_c = i / n_d;
    if (i_c_current != i_c) {
      relative_active_indices_list.emplace_back();
      i_c_current = i_c;
    }
    relative_active_indices_list.back().push_back(i % n_d);
  }
  return relative_active_indices_list;
}

/*
 * J_active_ad_list is a list of (n_d, n_v) matrices.
 * For QP contact dynamics, n_d is number of extreme rays in the polyhedral
 *  friction cone.
 * For SOCP contact dynamics, n_d is 3.
 *
 * For QP dynamics, for contact Jacobian in J_active_ad_list, it is possible
 *  that only some of its n_d rows are active. This is when the optional
 *  relative_active_indices_list becomes useful: for J_active_ad_list[i],
 *  relative_active_indices_list[i] stores the indices of its active rows,
 *  ranging from 0 to n_d - 1.
 *
 * This function returns DG_active_vecDq, a matrix of shape
 *  (n_lambda_active * n_v, n_q).
 *
 * NOTE THAT G_active = -J_active!!!
 */
template <Eigen::Index M>
MatrixXd CalcDGactiveDqFromJActiveList(
    const std::vector<Eigen::Matrix<AutoDiffXd, M, -1>>& J_active_ad_list,
    const std::vector<std::vector<int>>* relative_active_indices_list) {
  const int m = J_active_ad_list.front().rows();
  const auto n_v = J_active_ad_list.front().cols();
  const auto n_q = J_active_ad_list.front()(0, 0).derivatives().size();
  int n_la;  // Total number of active rows in G_active.
  if (relative_active_indices_list) {
    n_la = std::accumulate(relative_active_indices_list->begin(),
                           relative_active_indices_list->end(), 0,
                           [](int a, const std::vector<int>& b) {
                             return a + b.size();
                           });
  } else {
    n_la = J_active_ad_list.size() * m;
  }

  std::vector<int> row_indices_all(m);
  std::iota(row_indices_all.begin(), row_indices_all.end(), 0);

  MatrixXd DvecG_activeDq(n_la * n_v, n_q);
  for (int i_q = 0; i_q < n_q; i_q++) {
    // Fill one column of DvecG_activeDq.
    int i_G = 0;  // row index into DvecG_activeDq.
    for (int j = 0; j < n_v; j++) {
      for (int i_c = 0; i_c < J_active_ad_list.size(); i_c++) {
        const auto& J_i = J_active_ad_list[i_c];

        // Find indices of active rows of the current J_i.
        const std::vector<int>* row_indices{nullptr};
        if (relative_active_indices_list) {
          row_indices = &(relative_active_indices_list->at(i_c));
        } else {
          row_indices = &row_indices_all;
        }

        for (const auto& i : *row_indices) {
          DvecG_activeDq(i_G, i_q) = -J_i(i, j).derivatives()[i_q];
          i_G += 1;
        }
      }
    }
  }
  return DvecG_activeDq;
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxQp(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb,
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDe,
    const Eigen::Ref<const Eigen::MatrixXd>& Jn,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict, const double h,
    const size_t n_d) const {
  MatrixXd Dv_nextDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(Dv_nextDb, h, &Dv_nextDq);

  /*----------------------------------------------------------------*/
  // Compute Dv_nextDvecG from the KKT conditions of the QP.
  const auto& [Dv_nextDvecG_active, lambda_star_active_indices] =
      dqp_->get_DzDvecG_active();
  const auto n_la = lambda_star_active_indices.size();

  /*----------------------------------------------------------------*/
  // e := phi_constraints / h.
  MatrixXd De_active_Dq(n_la, n_q_);
  std::vector<int> active_contact_indices;
  for (int i = 0; i < n_la; i++) {
    const size_t i_c = lambda_star_active_indices[i] / n_d;
    De_active_Dq.row(i) = ConvertColVToQdot(q_dict, Jn.row(i_c)) / h;

    if (active_contact_indices.empty() ||
        active_contact_indices.back() != i_c) {
      active_contact_indices.push_back(i_c);
    }
  }

  Dv_nextDq += Dv_nextDe(Eigen::all, lambda_star_active_indices) * De_active_Dq;

  /*----------------------------------------------------------------*/
  if (!lambda_star_active_indices.empty()) {
    // This is skipped if there is no contact.
    // Compute DvecGDq using Autodiff through MBP.
    const auto q = GetQVecFromDict(q_dict);
    const auto q_ad = InitializeAutoDiff(q);
    UpdateMbpAdPositions(q_ad);
    const auto sdps_active =
        CalcSignedDistancePairsFromCollisionPairs(&active_contact_indices);
    // TODO(pang): only J_active_ad is used. Think of a less wasteful interface?
    std::vector<MatrixX<AutoDiffXd>> J_active_ad_list;
    MatrixX<AutoDiffXd> Jn_active_ad;
    VectorX<AutoDiffXd> phi_active_ad;
    cjc_ad_->CalcJacobianAndPhiQp(context_plant_ad_, sdps_active, n_d,
                                  &phi_active_ad, &Jn_active_ad,
                                  &J_active_ad_list);

    const auto relative_active_indices_list =
        CalcRelativeActiveIndicesList(lambda_star_active_indices, n_d);
    const auto DvecG_activeDq = CalcDGactiveDqFromJActiveList<-1>(
        J_active_ad_list, &relative_active_indices_list);

    Dv_nextDq += Dv_nextDvecG_active * DvecG_activeDq;
  }

  return CalcDq_nextDqFromDv_nextDq(Dv_nextDq, q_dict, v_star, h);
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxSocp(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb,
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDe,
    const std::vector<Eigen::Matrix3Xd>& J_list,
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict, double h) const {
  static constexpr int m{3};  // Dimension of 2nd order cones.

  MatrixXd Dv_nextDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(Dv_nextDb, h, &Dv_nextDq);

  const auto& [Dv_nextDvecG_active, lambda_star_active_indices] =
      dsocp_->get_DzDvecG_active();
  const auto n_la = lambda_star_active_indices.size();

  /*-------------------------------------------------------------------*/
  // e[i] := phi[i] / h / mu[i].
  MatrixXd De_active_Dq(n_la, n_q_);
  // The vector e, as defined in the SocpDerivatives class, e is an (m * n_l)
  // vector, where n_l == J_list.size(). But we know that for every m-length
  // segment of e, only the first element is a function of q.
  vector<int> active_indices_into_e;
  for (int i = 0; i < n_la; i++) {
    const int i_c = lambda_star_active_indices[i];
    De_active_Dq.row(i) = ConvertColVToQdot(q_dict, J_list[i_c].row(0)) / h;
    active_indices_into_e.push_back(i_c * m);
  }

  Dv_nextDq += Dv_nextDe(Eigen::all, active_indices_into_e) * De_active_Dq;
  /*----------------------------------------------------------------*/
  if (!lambda_star_active_indices.empty()) {
    const auto q = GetQVecFromDict(q_next_dict);
    UpdateMbpAdPositions(InitializeAutoDiff(q));
    const auto sdps_active =
        CalcSignedDistancePairsFromCollisionPairs(&lambda_star_active_indices);
    // TODO(pang): only J_active_ad is used. Think of a less wasteful interface?
    std::vector<Matrix3X<AutoDiffXd>> J_active_ad_list;
    VectorX<AutoDiffXd> phi_active_ad;
    cjc_ad_->CalcJacobianAndPhiSocp(context_plant_ad_, sdps_active,
                                    &phi_active_ad, &J_active_ad_list);

    const auto DvecG_activeDq =
        CalcDGactiveDqFromJActiveList<3>(J_active_ad_list, nullptr);

    Dv_nextDq += Dv_nextDvecG_active * DvecG_activeDq;
  }

  return CalcDq_nextDqFromDv_nextDq(Dv_nextDq, q_next_dict, v_star, h);
}

void QuasistaticSimulator::CalcDv_nextDbDq(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb, const double h,
    drake::EigenPtr<Eigen::MatrixXd> Dv_nextDq_ptr) const {
  MatrixXd DbDq = MatrixXd::Zero(n_v_, n_q_);
  for (const auto& model : models_actuated_) {
    const auto& idx_v = velocity_indices_.at(model);
    const auto& idx_q = position_indices_.at(model);
    const auto& Kq_i = robot_stiffness_.at(model);
    // TODO(pang): This needs double check!
    DbDq(idx_q, idx_v).diagonal() = h * Kq_i;
  }
  *Dv_nextDq_ptr += Dv_nextDb * DbDq;
}

Eigen::MatrixXd QuasistaticSimulator::CalcDq_nextDqFromDv_nextDq(
    const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDq,
    const ModelInstanceIndexToVecMap& q_dict,
    const Eigen::Ref<const Eigen::VectorXd>& v_star, const double h) const {
  if (n_v_ == n_q_) {
    return MatrixXd::Identity(n_v_, n_v_) + h * Dv_nextDq;
  }

  MatrixXd A = ConvertRowVToQdot(q_dict, Dv_nextDq);
  AddDNDq2A(v_star, &A);
  A *= h;
  A.diagonal() += VectorXd::Ones(n_q_);
  return A;
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxLogIcecream(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict, const double h,
    const double kappa, const Eigen::LLT<MatrixXd>& H_llt) const {
  MatrixXd DyDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(MatrixXd::Identity(n_v_, n_v_) * kappa, h, &DyDq);

  /*----------------------------------------------------------------*/
  const auto q = GetQVecFromDict(q_dict);
  const auto q_ad = InitializeAutoDiff(q);
  UpdateMbpAdPositions(q_ad);
  const auto sdps = CalcSignedDistancePairsFromCollisionPairs();
  const auto n_c = sdps.size();
  std::vector<Matrix3X<AutoDiffXd>> J_ad_list;
  VectorX<AutoDiffXd> phi_ad;
  cjc_ad_->CalcJacobianAndPhiSocp(context_plant_ad_, sdps, &phi_ad, &J_ad_list);

  //  cout << "DyDq\n" << DyDq << endl;

  VectorX<AutoDiffXd> y(n_v_);
  y.setZero();
  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& J = J_ad_list[i_c];
    Vector3<AutoDiffXd> w = J * v_star;
    w[0] += phi_ad[i_c] / h / cjc_->get_friction_coefficient(i_c);
    AutoDiffXd d = -w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    VectorX<AutoDiffXd> thing_to_add =
        2 * J.transpose() * Vector3<AutoDiffXd>(w[0] / d, -w[1] / d, -w[2] / d);
    y += thing_to_add;

    //    const auto A_to_add = drake::math::ExtractGradient(thing_to_add);
    //    cout << i_c;
    //    cout << " d " << d;
    //    cout << " phi " << phi_ad[i_c];
    //    cout << " A_max " << A_to_add.array().abs().maxCoeff();
    //    cout << "\n";
  }

  DyDq += drake::math::ExtractGradient(y);
  DyDq *= -1;
  H_llt.solveInPlace(DyDq);  // Now it becomes Dv_nextDq.

  return CalcDq_nextDqFromDv_nextDq(DyDq, q_next_dict, v_star, h);
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDxLogPyramid(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const ModelInstanceIndexToVecMap& q_dict,
    const ModelInstanceIndexToVecMap& q_next_dict,
    const QuasistaticSimParameters& params,
    const Eigen::LLT<Eigen::MatrixXd>& H_llt) const {
  const auto kappa = params.log_barrier_weight;
  const auto h = params.h;
  const auto n_d = params.nd_per_contact;

  MatrixXd DyDq = MatrixXd::Zero(n_v_, n_q_);
  CalcDv_nextDbDq(MatrixXd::Identity(n_v_, n_v_) * kappa, h, &DyDq);

  /*----------------------------------------------------------------*/
  const auto q = GetQVecFromDict(q_dict);
  UpdateMbpAdPositions(InitializeAutoDiff(q));
  const auto sdps = CalcSignedDistancePairsFromCollisionPairs();
  std::vector<MatrixX<AutoDiffXd>> J_ad_list;
  MatrixX<AutoDiffXd> Jn_ad;
  VectorX<AutoDiffXd> phi_ad;
  cjc_ad_->CalcJacobianAndPhiQp(context_plant_ad_, sdps, n_d, &phi_ad, &Jn_ad,
                                &J_ad_list);

  const auto n_c = sdps.size();
  VectorX<AutoDiffXd> y(n_v_);
  y.setZero();
  for (int i = 0; i < n_c; i++) {
    for (int j = 0; j < n_d; j++) {
      const Eigen::RowVectorX<AutoDiffXd>& J_ij = J_ad_list[i].row(j);
      const auto d = J_ij.dot(v_star) + phi_ad[i] / h;
      y -= J_ij.transpose() / d;
    }
  }

  DyDq += drake::math::ExtractGradient(y);
  DyDq *= -1;
  H_llt.solveInPlace(DyDq);  // Now it becomes Dv_nextDq.

  return CalcDq_nextDqFromDv_nextDq(DyDq, q_next_dict, v_star, h);
}

void QuasistaticSimulator::GetGeneralizedForceFromExternalSpatialForce(
    const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>&
        easf,
    ModelInstanceIndexToVecMap* tau_ext) const {
  // TODO(pang): actually process externally applied spatial force.
  if (!easf.empty()) {
    throw std::runtime_error(
        "Externally applied spatial force are not yet processed.");
  }
  for (const auto& model : models_actuated_) {
    (*tau_ext)[model] = Eigen::VectorXd::Zero(plant_->num_velocities(model));
  }
}

void QuasistaticSimulator::CalcGravityForUnactuatedModels(
    ModelInstanceIndexToVecMap* tau_ext) const {
  const auto gravity_all =
      plant_->CalcGravityGeneralizedForces(*context_plant_);

  for (const auto& model : models_unactuated_) {
    const auto& indices = velocity_indices_.at(model);
    const int n_v_i = indices.size();
    (*tau_ext)[model] = VectorXd(n_v_i);
    for (int i = 0; i < n_v_i; i++) {
      (*tau_ext)[model][i] = gravity_all[indices[i]];
    }
  }
}

ModelInstanceIndexToVecMap QuasistaticSimulator::CalcTauExt(
    const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>&
        easf_list) const {
  ModelInstanceIndexToVecMap tau_ext;
  GetGeneralizedForceFromExternalSpatialForce(easf_list, &tau_ext);
  CalcGravityForUnactuatedModels(&tau_ext);
  return tau_ext;
}

ModelInstanceNameToIndexMap
QuasistaticSimulator::GetModelInstanceNameToIndexMap() const {
  ModelInstanceNameToIndexMap name_to_index_map;
  for (const auto& model : models_all_) {
    name_to_index_map[plant_->GetModelInstanceName(model)] = model;
  }
  return name_to_index_map;
}

inline Eigen::Matrix3d MakeSkewSymmetricFromVec(
    const Eigen::Ref<const Vector3d>& v) {
  return Eigen::Matrix3d{{0, -v[2], v[1]}, {v[2], 0, -v[0]}, {-v[1], v[0], 0}};
}

Eigen::Matrix<double, 4, 3> QuasistaticSimulator::CalcNW2Qdot(
    const Eigen::Ref<const Eigen::Vector4d>& Q) {
  Eigen::Matrix<double, 4, 3> E;
  //  E.row(0) << -Q[1], -Q[2], -Q[3];
  //  E.row(1) << Q[0], Q[3], -Q[2];
  //  E.row(2) << -Q[3], Q[0], Q[1];
  //  E.row(3) << Q[2], -Q[1], Q[0];
  E.row(0) = -Q.tail(3);
  E.bottomRows(3) = -MakeSkewSymmetricFromVec(Q.tail(3));
  E.bottomRows(3).diagonal().setConstant(Q[0]);
  E *= 0.5;
  return E;
}

Eigen::Matrix<double, 3, 4> QuasistaticSimulator::CalcNQdot2W(
    const Eigen::Ref<const Eigen::Vector4d>& Q) {
  Eigen::Matrix<double, 3, 4> E;
  E.col(0) = -Q.tail(3);
  E.rightCols(3) = MakeSkewSymmetricFromVec(Q.tail(3));
  E.rightCols(3).diagonal().setConstant(Q[0]);
  E *= 2;
  return E;
}

Eigen::Map<const Eigen::VectorXi> QuasistaticSimulator::GetIndicesAsVec(
    const drake::multibody::ModelInstanceIndex& model,
    ModelIndicesMode mode) const {
  std::vector<int> const* indices{nullptr};
  if (mode == ModelIndicesMode::kQ) {
    indices = &position_indices_.at(model);
  } else {
    indices = &velocity_indices_.at(model);
  }

  return {indices->data(), static_cast<Eigen::Index>(indices->size())};
}

Eigen::MatrixXd QuasistaticSimulator::ConvertRowVToQdot(
    const ModelInstanceIndexToVecMap& q_dict,
    const Eigen::Ref<const Eigen::MatrixXd>& M_v) const {
  MatrixXd M_qdot(n_q_, M_v.cols());
  for (const auto& model : models_all_) {
    const auto idx_v_model = GetIndicesAsVec(model, ModelIndicesMode::kV);
    const auto idx_q_model = GetIndicesAsVec(model, ModelIndicesMode::kQ);

    if (is_model_floating(model)) {
      // If q contains a quaternion.
      const Eigen::Vector4d& Q_WB = q_dict.at(model).head(4);

      // Rotation.
      M_qdot(idx_q_model.head(4), Eigen::all) =
          CalcNW2Qdot(Q_WB) * M_v(idx_v_model.head(3), Eigen::all);
      // Translation.
      M_qdot(idx_q_model.tail(3), Eigen::all) =
          M_v(idx_v_model.tail(3), Eigen::all);
    } else {
      M_qdot(idx_q_model, Eigen::all) = M_v(idx_v_model, Eigen::all);
    }
  }

  return M_qdot;
}

Eigen::MatrixXd QuasistaticSimulator::ConvertColVToQdot(
    const ModelInstanceIndexToVecMap& q_dict,
    const Eigen::Ref<const Eigen::MatrixXd>& M_v) const {
  MatrixXd M_qdot(M_v.rows(), n_q_);
  for (const auto& model : models_all_) {
    const auto idx_v_model = GetIndicesAsVec(model, ModelIndicesMode::kV);
    const auto idx_q_model = GetIndicesAsVec(model, ModelIndicesMode::kQ);

    if (is_model_floating(model)) {
      const Eigen::Vector4d& Q_WB = q_dict.at(model).head(4);

      // Rotation.
      M_qdot(Eigen::all, idx_q_model.head(4)) =
          M_v(Eigen::all, idx_v_model.head(3)) * CalcNQdot2W(Q_WB);

      // Translation.
      M_qdot(Eigen::all, idx_q_model.tail(3)) =
          M_v(Eigen::all, idx_v_model.tail(3));
    } else {
      M_qdot(Eigen::all, idx_q_model) = M_v(Eigen::all, idx_v_model);
    }
  }

  return M_qdot;
}

void QuasistaticSimulator::AddDNDq2A(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    drake::EigenPtr<Eigen::MatrixXd> A_ptr) const {
  Eigen::Matrix4d E;
  for (const auto& model : models_unactuated_) {
    if (!is_model_floating(model)) {
      continue;
    }
    const auto idx_v_model = GetIndicesAsVec(model, ModelIndicesMode::kV);
    const auto idx_q_model = GetIndicesAsVec(model, ModelIndicesMode::kQ);
    const Vector3d& w = v_star(idx_v_model.head(3));  // angular velocity.

    E.row(0) << 0, -w[0], -w[1], -w[2];
    E.row(1) << w[0], 0, -w[2], w[1];
    E.row(2) << w[1], w[2], 0, -w[0];
    E.row(3) << w[2], -w[1], w[0], 0;

    (*A_ptr)(idx_q_model.head(4), idx_q_model.head(4)) += E;
  }
}

std::vector<drake::geometry::SignedDistancePair<drake::AutoDiffXd>>
QuasistaticSimulator::CalcSignedDistancePairsFromCollisionPairs(
    std::vector<int> const* active_contact_indices) const {
  std::vector<drake::geometry::SignedDistancePair<drake::AutoDiffXd>> sdps_ad;
  std::vector<int> all_indices;
  if (active_contact_indices == nullptr) {
    all_indices.resize(collision_pairs_.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    active_contact_indices = &all_indices;
  }

  for (const auto i : *active_contact_indices) {
    const auto& collision_pair = collision_pairs_[i];
    sdps_ad.push_back(query_object_ad_->ComputeSignedDistancePairClosestPoints(
        collision_pair.first, collision_pair.second));
  }
  return sdps_ad;
}

std::vector<drake::geometry::SignedDistancePair<double>>
QuasistaticSimulator::CalcCollisionPairs(
    double contact_detection_tolerance) const {
  auto sdps = query_object_->ComputeSignedDistancePairwiseClosestPoints(
      contact_detection_tolerance);
  collision_pairs_.clear();

  // Save collision pairs, which may later be used in gradient computation by
  // the AutoDiff MBP.
  for (const auto& sdp : sdps) {
    collision_pairs_.emplace_back(sdp.id_A, sdp.id_B);
  }
  return sdps;
}

ModelInstanceIndexToMatrixMap QuasistaticSimulator::CalcScaledMassMatrix(
    double h, double unactuated_mass_scale) const {
  MatrixXd M(n_v_, n_v_);
  plant_->CalcMassMatrix(*context_plant_, &M);

  ModelInstanceIndexToMatrixMap M_u_dict;
  for (const auto& model : models_unactuated_) {
    const auto& idx_v_model = velocity_indices_.at(model);
    M_u_dict[model] = M(idx_v_model, idx_v_model);
  }

  if (unactuated_mass_scale == 0 || std::isnan(unactuated_mass_scale)) {
    return M_u_dict;
  }

  std::unordered_map<drake::multibody::ModelInstanceIndex, double>
      max_eigen_value_M_u;
  for (const auto& model : models_unactuated_) {
    // TODO(pang): use the eigen value instead of maximum
    max_eigen_value_M_u[model] = M_u_dict.at(model).diagonal().maxCoeff();
  }

  const double min_K_a_h2 = min_K_a_ * h * h;

  for (auto& [model, M_u] : M_u_dict) {
    auto scale =
        min_K_a_h2 / max_eigen_value_M_u[model] / unactuated_mass_scale;
    M_u *= scale;
  }

  return M_u_dict;
}

void QuasistaticSimulator::UpdateQdictFromV(
    const Eigen::Ref<const Eigen::VectorXd>& v_star,
    const QuasistaticSimParameters& params,
    ModelInstanceIndexToVecMap* q_dict_ptr) const {
  const auto v_dict = GetVdictFromVec(v_star);
  auto& q_dict = *q_dict_ptr;
  const auto h = params.h;

  std::unordered_map<ModelInstanceIndex, VectorXd> dq_dict;
  for (const auto& model : models_all_) {
    const auto& idx_v = velocity_indices_.at(model);
    const auto n_q_i = plant_->num_positions(model);

    if (is_3d_floating_.at(model)) {
      // Positions of the model contains a quaternion. Conversion from
      // angular velocities to quaternion dot is necessary.
      const auto& q_u = q_dict[model];
      const Eigen::Vector4d Q(q_u.head(4));

      VectorXd dq_u(7);
      const auto& v_u = v_dict.at(model);
      dq_u.head(4) = CalcNW2Qdot(Q) * v_u.head(3) * h;
      dq_u.tail(3) = v_u.tail(3) * h;

      dq_dict[model] = dq_u;
    } else {
      dq_dict[model] = v_dict.at(model) * h;
    }
  }

  // TODO(pang): not updating unactuated object poses can lead to penetration at
  //  the next time step. A better solution is needed.
  if (params.unactuated_mass_scale > 0 ||
      std::isnan(params.unactuated_mass_scale)) {
    for (const auto& model : models_all_) {
      auto& q_model = q_dict[model];
      q_model += dq_dict[model];

      if (is_3d_floating_.at(model)) {
        // Normalize quaternion.
        q_model.head(4).normalize();
      }
    }
  } else {
    // un-actuated objects remain fixed.
    for (const auto& model : models_actuated_) {
      auto& q_model = q_dict[model];
      q_model += dq_dict[model];
    }
  }
}

VectorXd QuasistaticSimulator::CalcDynamics(
    QuasistaticSimulator* q_sim, const Eigen::Ref<const VectorXd>& q,
    const Eigen::Ref<const VectorXd>& u,
    const QuasistaticSimParameters& sim_params) {
  q_sim->UpdateMbpPositions(q);
  auto tau_ext_dict = q_sim->CalcTauExt({});
  auto q_a_cmd_dict = q_sim->GetQaCmdDictFromVec(u);
  q_sim->Step(q_a_cmd_dict, tau_ext_dict, sim_params);
  return q_sim->GetMbpPositionsAsVec();
}

VectorXd QuasistaticSimulator::CalcDynamics(
    const Eigen::Ref<const VectorXd>& q, const Eigen::Ref<const VectorXd>& u,
    const QuasistaticSimParameters& sim_params) {
  return CalcDynamics(this, q, u, sim_params);
}

std::unordered_map<drake::multibody::ModelInstanceIndex,
                   std::unordered_map<std::string, Eigen::VectorXd>>
QuasistaticSimulator::GetActuatedJointLimits() const {
  std::unordered_map<drake::multibody::ModelInstanceIndex,
                     std::unordered_map<std::string, Eigen::VectorXd>>
      joint_limits;
  for (const auto& model : models_actuated_) {
    const auto n_q = plant_->num_positions(model);
    joint_limits[model]["lower"] = Eigen::VectorXd(n_q);
    joint_limits[model]["upper"] = Eigen::VectorXd(n_q);
    int n_dofs = 0;
    for (const auto& joint_idx : plant_->GetJointIndices(model)) {
      const auto& joint = plant_->get_joint(joint_idx);
      const auto n_dof = joint.num_positions();
      if (n_dof != 1) {
        continue;
      }
      for (int j = 0; j < n_dof; j++) {
        auto lower = joint.position_lower_limits();
        auto upper = joint.position_upper_limits();
        joint_limits[model]["lower"][n_dofs] = lower[0];
        joint_limits[model]["upper"][n_dofs] = upper[0];
      }
      n_dofs += n_dof;
    }
    // No floating joints in the robots.
    DRAKE_THROW_UNLESS(n_q == n_dofs);
  }
  return joint_limits;
}

const drake::solvers::SolverInterface& QuasistaticSimulator::PickBestSocpSolver(
    const QuasistaticSimParameters& params) const {
  if (params.use_free_solvers) {
    return solver_selector_->get_solver(drake::solvers::ScsSolver::id());
  }
  // Commercial solvers.
  return solver_selector_->PickBestSocpSolver(is_socp_calculating_dual(params));
}

const drake::solvers::SolverInterface& QuasistaticSimulator::PickBestQpSolver(
    const QuasistaticSimParameters& params) const {
  if (params.use_free_solvers) {
    return solver_selector_->get_solver(drake::solvers::OsqpSolver::id());
  }
  return solver_selector_->PickBestQpSolver();
}

const drake::solvers::SolverInterface& QuasistaticSimulator::PickBestConeSolver(
    const QuasistaticSimParameters& params) const {
  if (params.use_free_solvers) {
    return solver_selector_->get_solver(drake::solvers::ScsSolver::id());
  }
  return solver_selector_->get_solver(drake::solvers::MosekSolver::id());
}

void QuasistaticSimulator::print_solver_info_for_default_params() const {
  const std::string socp_solver_name =
      PickBestConeSolver(sim_params_).solver_id().name();
  const std::string qp_solver_name =
      PickBestQpSolver(sim_params_).solver_id().name();
  const std::string cone_solver_name =
      PickBestConeSolver(sim_params_).solver_id().name();

  cout << "=========== Solver Info ===========" << endl;
  cout << "Using free solvers? " << sim_params_.use_free_solvers << endl;
  cout << "SOCP solver: " << socp_solver_name << endl;
  cout << "QP solver: " << qp_solver_name << endl;
  cout << "Cone solver: " << cone_solver_name << endl;
  cout << "===================================" << endl;
}
