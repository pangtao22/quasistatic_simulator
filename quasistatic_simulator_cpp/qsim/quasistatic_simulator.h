#pragma once
#include <iostream>

#include "diffcp/log_barrier_solver.h"
#include "diffcp/qp_derivatives.h"
#include "diffcp/socp_derivatives.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/scs_solver.h"
#include "qsim/contact_jacobian_calculator.h"

/*
 * Denotes whether the indices are those of a model's configuration vector
 * into the configuration vector of the system, or those of a model's velocity
 * vector into the velocity vector of the system.
 */
enum class ModelIndicesMode { kQ, kV };

class QuasistaticSimulator {
 public:
  QuasistaticSimulator(
      const std::string& model_directive_path,
      const std::unordered_map<std::string, Eigen::VectorXd>&
          robot_stiffness_str,
      const std::unordered_map<std::string, std::string>& object_sdf_paths,
      QuasistaticSimParameters sim_params);

  void UpdateMbpPositions(const ModelInstanceIndexToVecMap& q_dict);
  void UpdateMbpPositions(const Eigen::Ref<const Eigen::VectorXd>& q);
  // These methods are naturally const because context will eventually be
  // moved outside this class.
  void UpdateMbpAdPositions(const ModelInstanceIndexToVecAdMap& q_dict) const;
  void UpdateMbpAdPositions(
      const Eigen::Ref<const drake::AutoDiffVecXd>& q) const;

  [[nodiscard]] ModelInstanceIndexToVecMap GetMbpPositions() const;
  [[nodiscard]] Eigen::VectorXd GetMbpPositionsAsVec() const {
    return plant_->GetPositions(*context_plant_);
  }

  Eigen::VectorXd GetPositions(
      drake::multibody::ModelInstanceIndex model) const;

  void Step(const ModelInstanceIndexToVecMap& q_a_cmd_dict,
            const ModelInstanceIndexToVecMap& tau_ext_dict,
            const QuasistaticSimParameters& params);

  void Step(const ModelInstanceIndexToVecMap& q_a_cmd_dict,
            const ModelInstanceIndexToVecMap& tau_ext_dict);

  void GetGeneralizedForceFromExternalSpatialForce(
      const std::vector<
          drake::multibody::ExternallyAppliedSpatialForce<double>>& easf,
      ModelInstanceIndexToVecMap* tau_ext) const;

  void CalcGravityForUnactuatedModels(
      ModelInstanceIndexToVecMap* tau_ext) const;

  ModelInstanceIndexToVecMap CalcTauExt(
      const std::vector<
          drake::multibody::ExternallyAppliedSpatialForce<double>>& easf_list)
      const;

  ModelInstanceNameToIndexMap GetModelInstanceNameToIndexMap() const;

  [[nodiscard]] const std::set<drake::multibody::ModelInstanceIndex>&
  get_all_models() const {
    return models_all_;
  }

  [[nodiscard]] const std::set<drake::multibody::ModelInstanceIndex>&
  get_actuated_models() const {
    return models_actuated_;
  }

  [[nodiscard]] const std::set<drake::multibody::ModelInstanceIndex>&
  get_unactuated_models() const {
    return models_unactuated_;
  }

  [[nodiscard]] const QuasistaticSimParameters& get_sim_params() const {
    return sim_params_;
  }

  [[nodiscard]] QuasistaticSimParameters& get_mutable_sim_params() {
    return sim_params_;
  }

  [[nodiscard]] QuasistaticSimParameters get_sim_params_copy() const {
    return sim_params_;
  }

  /*
   * Technically it only makes sense for Bodies to float. But our convention
   * is that un-actuated model instances only consist of one rigid body.
   */
  [[nodiscard]] bool is_model_floating(
      drake::multibody::ModelInstanceIndex m) const {
    return is_3d_floating_.at(m);
  }

  const drake::geometry::QueryObject<double>& get_query_object() const {
    return *query_object_;
  }

  const drake::multibody::MultibodyPlant<double>& get_plant() const {
    return *plant_;
  }

  const drake::geometry::SceneGraph<double>& get_scene_graph() const {
    return *sg_;
  }

  const drake::systems::Diagram<double>& get_diagram() const {
    return *diagram_;
  }

  const drake::geometry::QueryObject<drake::AutoDiffXd>& get_query_object_ad()
      const {
    return *query_object_ad_;
  }

  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& get_plant_ad()
      const {
    return *plant_ad_;
  }

  const drake::geometry::SceneGraph<drake::AutoDiffXd>& get_scene_graph_ad()
      const {
    return *sg_ad_;
  }

  const drake::systems::Diagram<drake::AutoDiffXd>& get_diagram_ad() const {
    return *diagram_ad_;
  }

  const drake::multibody::ContactResults<double>& get_contact_results() const {
    return contact_results_;
  }

  drake::multibody::ContactResults<double> GetContactResultsCopy() const {
    return contact_results_;
  }

  void update_sim_params(const QuasistaticSimParameters& new_params) {
    sim_params_ = new_params;
  }

  int num_actuated_dofs() const { return n_v_a_; }
  int num_unactuated_dofs() const { return n_v_u_; }
  int num_dofs() const { return n_v_a_ + n_v_u_; }

  Eigen::MatrixXd get_Dq_nextDq() const { return Dq_nextDq_; }
  Eigen::MatrixXd get_Dq_nextDqa_cmd() const { return Dq_nextDqa_cmd_; }

  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
  GetVelocityIndices() const {
    return velocity_indices_;
  }

  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
  GetPositionIndices() const {
    return position_indices_;
  }

  ModelInstanceIndexToVecMap GetVdictFromVec(
      const Eigen::Ref<const Eigen::VectorXd>& v) const;

  ModelInstanceIndexToVecMap GetQDictFromVec(
      const Eigen::Ref<const Eigen::VectorXd>& q) const;

  Eigen::VectorXd GetQVecFromDict(
      const ModelInstanceIndexToVecMap& q_dict) const;

  /*
   * QaCmd, sometimes denoted by u, is the concatenation of position vectors
   * for all models in this->models_actuated_, which is sorted in ascending
   * order.
   * They keys of q_a_cmd_dict does not need to be the same as
   * this->models_actuated. It only needs to be a superset of
   * this->models_actuated. This means that it is possible to pass in a
   * dictionary containing position vectors for all model instances in the
   * system, including potentially position vectors of un-actuated models,
   * and this method will extract the actuated position vectors and
   * concatenate them into a single vector.
   */
  Eigen::VectorXd GetQaCmdVecFromDict(
      const ModelInstanceIndexToVecMap& q_a_cmd_dict) const;

  ModelInstanceIndexToVecMap GetQaCmdDictFromVec(
      const Eigen::Ref<const Eigen::VectorXd>& q_a_cmd) const;

  Eigen::VectorXi GetQaIndicesIntoQ() const;
  Eigen::VectorXi GetQuIndicesIntoQ() const;
  Eigen::VectorXi GetModelsIndicesIntoQ(
      const std::set<drake::multibody::ModelInstanceIndex>& models) const;

  static Eigen::VectorXd CalcDynamics(
      QuasistaticSimulator* q_sim, const Eigen::Ref<const Eigen::VectorXd>& q,
      const Eigen::Ref<const Eigen::VectorXd>& u,
      const QuasistaticSimParameters& sim_params);

  /*
   * Wrapper around QuasistaticSimulator::Step, which takes as inputs state
   * and input vectors, and returns the next state as a vector.
   */
  Eigen::VectorXd CalcDynamics(const Eigen::Ref<const Eigen::VectorXd>& q,
                               const Eigen::Ref<const Eigen::VectorXd>& u,
                               const QuasistaticSimParameters& sim_params);

  Eigen::MatrixXd ConvertRowVToQdot(
      const ModelInstanceIndexToVecMap& q_dict,
      const Eigen::Ref<const Eigen::MatrixXd>& M_v) const;

  Eigen::MatrixXd ConvertColVToQdot(
      const ModelInstanceIndexToVecMap& q_dict,
      const Eigen::Ref<const Eigen::MatrixXd>& M_v) const;

  ModelInstanceIndexToMatrixMap CalcScaledMassMatrix(
      double h, double unactuated_mass_scale) const;

  std::unordered_map<drake::multibody::ModelInstanceIndex,
                     std::unordered_map<std::string, Eigen::VectorXd>>
  GetActuatedJointLimits() const;

  void print_solver_info_for_default_params() const;

 private:
  static Eigen::Matrix<double, 4, 3> CalcNW2Qdot(
      const Eigen::Ref<const Eigen::Vector4d>& Q);

  static Eigen::Matrix<double, 3, 4> CalcNQdot2W(
      const Eigen::Ref<const Eigen::Vector4d>& Q);

  Eigen::Map<const Eigen::VectorXi> GetIndicesAsVec(
      const drake::multibody::ModelInstanceIndex& model,
      ModelIndicesMode mode) const;

  void AddDNDq2A(const Eigen::Ref<const Eigen::VectorXd>& v_star,
                 drake::EigenPtr<Eigen::MatrixXd> A_ptr) const;

  [[nodiscard]] std::vector<int> GetIndicesForModel(
      drake::multibody::ModelInstanceIndex idx, ModelIndicesMode mode) const;

  void CalcQAndTauH(const ModelInstanceIndexToVecMap& q_dict,
                    const ModelInstanceIndexToVecMap& q_a_cmd_dict,
                    const ModelInstanceIndexToVecMap& tau_ext_dict, double h,
                    Eigen::MatrixXd* Q_ptr, Eigen::VectorXd* tau_h_ptr,
                    double unactuated_mass_scale) const;

  Eigen::MatrixXd CalcDfDu(const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb,
                           double h,
                           const ModelInstanceIndexToVecMap& q_dict) const;

  Eigen::MatrixXd CalcDfDxQp(const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb,
                             const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDe,
                             const Eigen::Ref<const Eigen::MatrixXd>& Jn,
                             const Eigen::Ref<const Eigen::VectorXd>& v_star,
                             const ModelInstanceIndexToVecMap& q_dict, double h,
                             size_t n_d) const;

  Eigen::MatrixXd CalcDfDxSocp(
      const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb,
      const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDe,
      const std::vector<Eigen::Matrix3Xd>& J_list,
      const Eigen::Ref<const Eigen::VectorXd>& v_star,
      const ModelInstanceIndexToVecMap& q_dict,
      const ModelInstanceIndexToVecMap& q_next_dict, double h) const;
  /*
   * Adds Dv_nextDb * DbDq to Dv_nextDq.
   */
  void CalcDv_nextDbDq(const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDb,
                       double h,
                       drake::EigenPtr<Eigen::MatrixXd> Dv_nextDq_ptr) const;

  Eigen::MatrixXd CalcDq_nextDqFromDv_nextDq(
      const Eigen::Ref<const Eigen::MatrixXd>& Dv_nextDq,
      const ModelInstanceIndexToVecMap& q_dict,
      const Eigen::Ref<const Eigen::VectorXd>& v_star, double h) const;

  Eigen::MatrixXd CalcDfDxLogIcecream(
      const Eigen::Ref<const Eigen::VectorXd>& v_star,
      const ModelInstanceIndexToVecMap& q_dict,
      const ModelInstanceIndexToVecMap& q_next_dict, double h, double kappa,
      const Eigen::LLT<Eigen::MatrixXd>& H_llt) const;

  Eigen::MatrixXd CalcDfDxLogPyramid(
      const Eigen::Ref<const Eigen::VectorXd>& v_star,
      const ModelInstanceIndexToVecMap& q_dict,
      const ModelInstanceIndexToVecMap& q_next_dict,
      const QuasistaticSimParameters& params,
      const Eigen::LLT<Eigen::MatrixXd>& H_llt) const;

  void CalcUnconstrainedBFromHessian(const Eigen::LLT<Eigen::MatrixXd>& H_llt,
                                     const QuasistaticSimParameters& params,
                                     const ModelInstanceIndexToVecMap& q_dict,
                                     Eigen::MatrixXd* B_ptr) const;

  std::vector<drake::geometry::SignedDistancePair<double>> CalcCollisionPairs(
      double contact_detection_tolerance) const;

  std::vector<drake::geometry::SignedDistancePair<drake::AutoDiffXd>>
  CalcSignedDistancePairsFromCollisionPairs(
      std::vector<int> const* active_contact_indices = nullptr) const;

  void UpdateQdictFromV(const Eigen::Ref<const Eigen::VectorXd>& v_star,
                        const QuasistaticSimParameters& params,
                        ModelInstanceIndexToVecMap* q_dict_ptr) const;

  void CalcPyramidMatrices(const ModelInstanceIndexToVecMap& q_dict,
                           const ModelInstanceIndexToVecMap& q_a_cmd_dict,
                           const ModelInstanceIndexToVecMap& tau_ext_dict,
                           const QuasistaticSimParameters& params,
                           Eigen::MatrixXd* Q, Eigen::VectorXd* tau_h_ptr,
                           Eigen::MatrixXd* Jn_ptr, Eigen::MatrixXd* J_ptr,
                           Eigen::VectorXd* phi_ptr,
                           Eigen::VectorXd* phi_constraints_ptr) const;

  void CalcIcecreamMatrices(const ModelInstanceIndexToVecMap& q_dict,
                            const ModelInstanceIndexToVecMap& q_a_cmd_dict,
                            const ModelInstanceIndexToVecMap& tau_ext_dict,
                            const QuasistaticSimParameters& params,
                            Eigen::MatrixXd* Q, Eigen::VectorXd* tau_h,
                            std::vector<Eigen::Matrix3Xd>* J_list,
                            Eigen::VectorXd* phi) const;

  static void CalcContactResultsQp(
      const std::vector<ContactPairInfo<double>>& contact_info_list,
      const Eigen::Ref<const Eigen::VectorXd>& beta_star, int n_d, double h,
      drake::multibody::ContactResults<double>* contact_results);

  static void CalcContactResultsSocp(
      const std::vector<ContactPairInfo<double>>& contact_info_list,
      const std::vector<Eigen::VectorXd>& lambda_star, double h,
      drake::multibody::ContactResults<double>* contact_results);

  void ForwardQp(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                 const Eigen::Ref<const Eigen::VectorXd>& tau_h,
                 const Eigen::Ref<const Eigen::MatrixXd>& J,
                 const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
                 const QuasistaticSimParameters& params,
                 ModelInstanceIndexToVecMap* q_dict_ptr,
                 Eigen::VectorXd* v_star_ptr, Eigen::VectorXd* beta_star_ptr);

  void ForwardSocp(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                   const Eigen::Ref<const Eigen::VectorXd>& tau_h,
                   const std::vector<Eigen::Matrix3Xd>& J_list,
                   const Eigen::Ref<const Eigen::VectorXd>& phi,
                   const QuasistaticSimParameters& params,
                   ModelInstanceIndexToVecMap* q_dict_ptr,
                   Eigen::VectorXd* v_star_ptr,
                   std::vector<Eigen::VectorXd>* lambda_star_ptr,
                   std::vector<Eigen::VectorXd>* e_list);

  void ForwardLogPyramid(
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::VectorXd>& tau_h,
      const Eigen::Ref<const Eigen::MatrixXd>& J,
      const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
      const QuasistaticSimParameters& params,
      ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr);

  void ForwardLogPyramidInHouse(
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::VectorXd>& tau_h,
      const Eigen::Ref<const Eigen::MatrixXd>& J,
      const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
      const QuasistaticSimParameters& params,
      ModelInstanceIndexToVecMap* q_dict_ptr, Eigen::VectorXd* v_star_ptr);

  void ForwardLogIcecream(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                          const Eigen::Ref<const Eigen::VectorXd>& tau_h,
                          const std::vector<Eigen::Matrix3Xd>& J_list,
                          const Eigen::Ref<const Eigen::VectorXd>& phi,
                          const QuasistaticSimParameters& params,
                          ModelInstanceIndexToVecMap* q_dict_ptr,
                          Eigen::VectorXd* v_star_ptr);

  void BackwardQp(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                  const Eigen::Ref<const Eigen::VectorXd>& tau_h,
                  const Eigen::Ref<const Eigen::MatrixXd>& Jn,
                  const Eigen::Ref<const Eigen::MatrixXd>& J,
                  const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
                  const ModelInstanceIndexToVecMap& q_dict,
                  const ModelInstanceIndexToVecMap& q_dict_next,
                  const Eigen::Ref<const Eigen::VectorXd>& v_star,
                  const Eigen::Ref<const Eigen::VectorXd>& lambda_star,
                  const QuasistaticSimParameters& params);

  void BackwardSocp(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                    const Eigen::Ref<const Eigen::VectorXd>& tau_h,
                    const std::vector<Eigen::Matrix3Xd>& J_list,
                    const std::vector<Eigen::VectorXd>& e_list,
                    const Eigen::Ref<const Eigen::VectorXd>& phi,
                    const ModelInstanceIndexToVecMap& q_dict,
                    const ModelInstanceIndexToVecMap& q_dict_next,
                    const Eigen::Ref<const Eigen::VectorXd>& v_star,
                    const std::vector<Eigen::VectorXd>& lambda_star_list,
                    const QuasistaticSimParameters& params);

  void BackwardLogPyramid(
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& J,
      const Eigen::Ref<const Eigen::VectorXd>& phi_constraints,
      const ModelInstanceIndexToVecMap& q_dict,
      const ModelInstanceIndexToVecMap& q_next_dict,
      const Eigen::Ref<const Eigen::VectorXd>& v_star,
      const QuasistaticSimParameters& params,
      Eigen::LLT<Eigen::MatrixXd> const* const H_llt);

  void BackwardLogIcecream(const ModelInstanceIndexToVecMap& q_dict,
                           const ModelInstanceIndexToVecMap& q_next_dict,
                           const Eigen::Ref<const Eigen::VectorXd>& v_star,
                           const QuasistaticSimParameters& params,
                           const Eigen::LLT<Eigen::MatrixXd>& H_llt);

  bool is_socp_calculating_dual(const QuasistaticSimParameters& params) const {
    return params.calc_contact_forces ||
           (params.gradient_mode != GradientMode::kNone);
  }
  drake::solvers::SolverBase* PickBestSocpSolver(
      const QuasistaticSimParameters& params) const;

  drake::solvers::SolverBase* PickBestQpSolver(
      const QuasistaticSimParameters& params) const;

  drake::solvers::SolverBase* PickBestConeSolver(
      const QuasistaticSimParameters& params) const;

  QuasistaticSimParameters sim_params_;

  // Solvers.
  std::unique_ptr<drake::solvers::ScsSolver> solver_scs_;
  std::unique_ptr<drake::solvers::OsqpSolver> solver_osqp_;
  std::unique_ptr<drake::solvers::GurobiSolver> solver_grb_;
  std::unique_ptr<drake::solvers::MosekSolver> solver_msk_;
  std::unique_ptr<QpLogBarrierSolver> solver_log_pyramid_;
  std::unique_ptr<SocpLogBarrierSolver> solver_log_icecream_;
  mutable drake::solvers::MathematicalProgramResult mp_result_;

  // Optimization derivatives. Refer to the python implementation of
  //  QuasistaticSimulator for more details.
  std::unique_ptr<QpDerivativesActive> dqp_;
  std::unique_ptr<SocpDerivatives> dsocp_;
  Eigen::MatrixXd Dq_nextDq_;
  Eigen::MatrixXd Dq_nextDqa_cmd_;

  // Systems.
  std::unique_ptr<drake::systems::Diagram<double>> diagram_;
  drake::multibody::MultibodyPlant<double>* plant_{nullptr};
  drake::geometry::SceneGraph<double>* sg_{nullptr};

  // AutoDiff Systems.
  std::unique_ptr<drake::systems::Diagram<drake::AutoDiffXd>> diagram_ad_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>* plant_ad_{nullptr};
  const drake::geometry::SceneGraph<drake::AutoDiffXd>* sg_ad_{nullptr};

  // Contexts.
  std::unique_ptr<drake::systems::Context<double>> context_;  // Diagram.
  drake::systems::Context<double>* context_plant_{nullptr};
  drake::systems::Context<double>* context_sg_{nullptr};

  // AutoDiff contexts
  std::unique_ptr<drake::systems::Context<drake::AutoDiffXd>> context_ad_;
  mutable drake::systems::Context<drake::AutoDiffXd>* context_plant_ad_{
      nullptr};
  mutable drake::systems::Context<drake::AutoDiffXd>* context_sg_ad_{nullptr};

  // Internal state (for interfacing with QuasistaticSystem).
  const drake::geometry::QueryObject<double>* query_object_{nullptr};
  mutable std::vector<CollisionPair> collision_pairs_;
  mutable const drake::geometry::QueryObject<drake::AutoDiffXd>*
      query_object_ad_{nullptr};
  mutable drake::multibody::ContactResults<double> contact_results_;

  // MBP introspection.
  int n_v_a_{0};  // number of actuated DOFs.
  int n_v_u_{0};  // number of un-actuated DOFs.
  int n_v_{0};    // total number of velocities.
  int n_q_{0};    // total number of positions.
  std::set<drake::multibody::ModelInstanceIndex> models_actuated_;
  std::set<drake::multibody::ModelInstanceIndex> models_unactuated_;
  std::set<drake::multibody::ModelInstanceIndex> models_all_;
  std::unordered_map<drake::multibody::ModelInstanceIndex, bool>
      is_3d_floating_;
  ModelInstanceIndexToVecMap robot_stiffness_;
  double min_K_a_{0};  // smallest stiffness of all joints.
  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
      velocity_indices_;
  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
      position_indices_;

  std::unique_ptr<ContactJacobianCalculator<double>> cjc_;
  std::unique_ptr<ContactJacobianCalculator<drake::AutoDiffXd>> cjc_ad_;
};
