#include "qsim/contact_jacobian_calculator.h"

#include <iostream>

using drake::AutoDiffXd;
using drake::Matrix3X;
using drake::MatrixX;
using drake::Vector3;
using drake::Vector4;
using drake::VectorX;
using drake::multibody::ModelInstanceIndex;
using std::vector;

using std::cout;
using std::endl;

template <class T>
drake::Matrix3X<T> CalcTangentVectors(const Vector3<T>& normal,
                                      const size_t nd) {
  Vector3<T> n = normal.normalized();
  Vector4<T> n4(n.x(), n.y(), n.z(), 0);
  Matrix3X<T> tangents(3, nd);
  if (nd == 2) {
    // Makes sure that dC is in the yz plane.
    Vector4<T> n_x4(1, 0, 0, 0);
    tangents.col(0) = n_x4.cross3(n4).head(3);
    tangents.col(1) = -tangents.col(0);
  } else {
    const auto R = drake::math::RotationMatrix<T>::MakeFromOneUnitVector(n, 2);

    for (int i = 0; i < nd; i++) {
      const double theta = 2 * M_PI / nd * i;
      tangents.col(i) << cos(theta), sin(theta), 0;
    }
    tangents = R * tangents;
  }

  return tangents;
}

template <class T>
ContactJacobianCalculator<T>::ContactJacobianCalculator(
    const drake::systems::Diagram<T>* diagram,
    std::set<drake::multibody::ModelInstanceIndex> models_all)
    : models_all_(std::move(models_all)) {
  plant_ = dynamic_cast<const drake::multibody::MultibodyPlant<T>*>(
      &diagram->GetSubsystemByName(kMultiBodyPlantName));
  sg_ = dynamic_cast<const drake::geometry::SceneGraph<T>*>(
      &diagram->GetSubsystemByName(kSceneGraphName));

  DRAKE_THROW_UNLESS(plant_ != nullptr);
  DRAKE_THROW_UNLESS(sg_ != nullptr);

  // friction coefficients.
  const auto& inspector = sg_->model_inspector();
  const auto cc = inspector.GetCollisionCandidates();
  for (const auto& [g_idA, g_idB] : cc) {
    const double mu = GetFrictionCoefficientForSignedDistancePair(g_idA, g_idB);
    friction_coefficients_[g_idA][g_idB] = mu;
    friction_coefficients_[g_idB][g_idA] = mu;
  }
}

template <class T>
Matrix3X<T> ContactJacobianCalculator<T>::CalcContactJaocibanFromPoint(
    const drake::systems::Context<T>* context_plant,
    const drake::multibody::BodyIndex& body_idx,
    const drake::VectorX<T>& pC_Body) const {
  const auto& frameB = plant_->get_body(body_idx).body_frame();
  drake::Matrix3X<T> J_body(3, plant_->num_velocities());
  plant_->CalcJacobianTranslationalVelocity(
      *context_plant, drake::multibody::JacobianWrtVariable::kV, frameB,
      pC_Body, plant_->world_frame(), plant_->world_frame(), &J_body);

  return J_body;
}

template <class T>
void ContactJacobianCalculator<T>::UpdateContactPairInfo(
    const drake::systems::Context<T>* context_plant,
    const std::vector<drake::geometry::SignedDistancePair<T>>& sdps) const {
  // Clear stored contact info.
  contact_pairs_.clear();

  const auto n_c = sdps.size();
  const int n_v = plant_->num_velocities();
  const auto& inspector = sg_->model_inspector();

  contact_pairs_.resize(n_c);

  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& sdp = sdps[i_c];
    auto& cpi = contact_pairs_[i_c];  // ContactPairInfo.
    cpi.nhat_BA_W = sdp.nhat_BA_W;
    cpi.mu = friction_coefficients_.at(sdp.id_A).at(sdp.id_B);

    // Compute contact Jacobian.
    cpi.Jc.resize(3, n_v);
    cpi.Jc.setZero();
    const auto bodyA_idx = GetMbpBodyFromGeometry(sdp.id_A);
    const auto bodyB_idx = GetMbpBodyFromGeometry(sdp.id_B);
    const auto& X_AGa = inspector.GetPoseInFrame(sdp.id_A).template cast<T>();
    const auto& X_AGb = inspector.GetPoseInFrame(sdp.id_B).template cast<T>();
    const auto p_ACa_A = X_AGa * sdp.p_ACa;
    const auto p_BCb_B = X_AGb * sdp.p_BCb;
    const auto model_A_ptr = FindModelForBody(bodyA_idx);
    const auto model_B_ptr = FindModelForBody(bodyB_idx);

    if (model_A_ptr == nullptr && model_B_ptr == nullptr) {
      throw std::logic_error(
          "One body in a contact pair is not in body_indices_");
    } else {
      if (model_A_ptr) {
        cpi.Jc +=
            CalcContactJaocibanFromPoint(context_plant, bodyA_idx, p_ACa_A);
      }
      if (model_B_ptr) {
        cpi.Jc -=
            CalcContactJaocibanFromPoint(context_plant, bodyB_idx, p_BCb_B);
      }
    }

    // This is needed only for computing the contact point, which is now only
    // used for visualization.
    auto X_WA = plant_->EvalBodyPoseInWorld(*context_plant,
                                            plant_->get_body(bodyA_idx));
    auto X_WB = plant_->EvalBodyPoseInWorld(*context_plant,
                                            plant_->get_body(bodyB_idx));
    cpi.p_WCa = X_WA * p_ACa_A;
    cpi.p_WCb = X_WB * p_BCb_B;
    cpi.body_A_idx = bodyA_idx;
    cpi.body_B_idx = bodyB_idx;
    cpi.id_A = sdp.id_A;
    cpi.id_B = sdp.id_B;
  }
}

/*
 * Returns nullptr if body_idx is not in any of the values of bodies_indices_;
 * Otherwise returns the model instance to which body_idx belongs.
 */
template <class T>
std::unique_ptr<ModelInstanceIndex>
ContactJacobianCalculator<T>::FindModelForBody(
    drake::multibody::BodyIndex body_idx) const {
  const auto& model = plant_->get_body(body_idx).model_instance();
  if (models_all_.find(model) != models_all_.end()) {
    return std::make_unique<ModelInstanceIndex>(model);
  }
  return nullptr;
}

template <class T>
double
ContactJacobianCalculator<T>::GetFrictionCoefficientForSignedDistancePair(
    drake::geometry::GeometryId id_A, drake::geometry::GeometryId id_B) const {
  const auto& inspector = sg_->model_inspector();
  const auto props_A = inspector.GetProximityProperties(id_A);
  const auto props_B = inspector.GetProximityProperties(id_B);
  const auto& geometryA_friction =
      props_A->template GetProperty<drake::multibody::CoulombFriction<double>>(
          "material", "coulomb_friction");
  const auto& geometryB_friction =
      props_B->template GetProperty<drake::multibody::CoulombFriction<double>>(
          "material", "coulomb_friction");
  auto cf = drake::multibody::CalcContactFrictionFromSurfaceProperties(
      geometryA_friction, geometryB_friction);
  return cf.static_friction();
}

template <class T>
drake::multibody::BodyIndex
ContactJacobianCalculator<T>::GetMbpBodyFromGeometry(
    drake::geometry::GeometryId g_id) const {
  const auto& inspector = sg_->model_inspector();
  return plant_->GetBodyFromFrameId(inspector.GetFrameId(g_id))->index();
}

template <class T>
void ContactJacobianCalculator<T>::CalcJacobianAndPhiQp(
    const drake::systems::Context<T>* context_plant,
    const vector<drake::geometry::SignedDistancePair<T>>& sdps, const int n_d,
    drake::VectorX<T>* phi_ptr, drake::MatrixX<T>* Jn_ptr,
    std::vector<drake::MatrixX<T>>* J_list_ptr) const {
  UpdateContactPairInfo(context_plant, sdps);

  const auto n_c = sdps.size();
  const int n_v = plant_->num_velocities();
  const auto n_f = n_d * n_c;

  VectorX<T>& phi = *phi_ptr;
  MatrixX<T>& Jn = *Jn_ptr;
  phi.resize(n_c);
  Jn.resize(n_c, n_v);
  J_list_ptr->clear();

  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& sdp = sdps[i_c];
    const auto& cpi = contact_pairs_[i_c];
    const auto mu = get_friction_coefficient(i_c);

    phi[i_c] = sdp.distance;
    Jn.row(i_c) = sdp.nhat_BA_W.transpose() * cpi.Jc;

    contact_pairs_[i_c].t_W = CalcTangentVectors<T>(sdp.nhat_BA_W, n_d);
    const auto& d_W = contact_pairs_[i_c].t_W;
    J_list_ptr->template emplace_back(n_d, n_v);
    MatrixX<T>& J_i_c = J_list_ptr->back();
    for (int j = 0; j < n_d; j++) {
      J_i_c.row(j) = Jn.row(i_c) + mu * d_W.col(j).transpose() * cpi.Jc;
    }
  }
}

template <class T>
void ContactJacobianCalculator<T>::CalcJacobianAndPhiSocp(
    const drake::systems::Context<T>* context_plant,
    const std::vector<drake::geometry::SignedDistancePair<T>>& sdps,
    drake::VectorX<T>* phi_ptr,
    std::vector<drake::Matrix3X<T>>* J_list_ptr) const {
  UpdateContactPairInfo(context_plant, sdps);

  const auto n_c = sdps.size();
  const int n_v = plant_->num_velocities();

  VectorX<T>& phi = *phi_ptr;
  std::vector<drake::Matrix3X<T>>& J_list = *J_list_ptr;
  phi.resize(n_c);
  J_list.clear();

  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto& sdp = sdps[i_c];
    const auto& cpi = contact_pairs_[i_c];
    const auto mu = get_friction_coefficient(i_c);

    phi[i_c] = sdp.distance;
    J_list.template emplace_back(3, n_v);
    Matrix3X<T>& J_i = J_list.back();
    const drake::Matrix3<T> R =
        drake::math::RotationMatrix<T>::MakeFromOneUnitVector(sdp.nhat_BA_W, 2)
            .matrix();
    const Vector3<T>& t1 = R.col(0);
    const Vector3<T>& t2 = R.col(1);
    contact_pairs_[i_c].t_W.resize(3, 2);
    contact_pairs_[i_c].t_W.col(0) = t1;
    contact_pairs_[i_c].t_W.col(1) = t2;

    J_i.row(0) = sdp.nhat_BA_W.transpose() * cpi.Jc / mu;
    J_i.row(1) = t1.transpose() * cpi.Jc;
    J_i.row(2) = t2.transpose() * cpi.Jc;
  }
}

template class ContactJacobianCalculator<double>;
template class ContactJacobianCalculator<drake::AutoDiffXd>;
