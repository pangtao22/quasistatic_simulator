#include <string>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include "drake/multibody/plant/multibody_plant.h"

using ModelInstanceIndexToVecMap =
    std::unordered_map<drake::multibody::ModelInstanceIndex, Eigen::VectorXd>;
using ModelInstanceIndexToVecAdMap =
    std::unordered_map<drake::multibody::ModelInstanceIndex,
                       drake::AutoDiffVecXd>;
using ModelInstanceIndexToMatrixMap =
    std::unordered_map<drake::multibody::ModelInstanceIndex, Eigen::MatrixXd>;
using ModelInstanceNameToIndexMap =
    std::unordered_map<std::string, drake::multibody::ModelInstanceIndex>;
using CollisionPair =
    std::pair<drake::geometry::GeometryId, drake::geometry::GeometryId>;

/*
 * Gradient computation mode of QuasistaticSimulator.
 * Using an analogy from torch, GradientMode is the mode when "backward()" is
 *  called, after the forward dynamics is done.
 * - kNone: do not compute gradient, just roll out the dynamics.
 * - kBOnly: only computes dfdu, where x_next = f(x, u).
 * - kAB: computes both dfdx and dfdu.
 */
enum class GradientMode { kNone, kBOnly, kAB };

enum class ForwardDynamicsMode {
  kQpMp,
  kQpCvx,
  kSocpMp,
  kLogPyramidMp,
  kLogPyramidCvx,
  kLogPyramidMy,
  kLogIcecream
};

static const std::unordered_set<ForwardDynamicsMode> kPyramidModes{
    ForwardDynamicsMode::kQpMp, ForwardDynamicsMode::kLogPyramidMp,
    ForwardDynamicsMode::kLogPyramidMy};

static const std::unordered_set<ForwardDynamicsMode> kIcecreamModes{
  ForwardDynamicsMode::kSocpMp, ForwardDynamicsMode::kLogIcecream};

/*
h: simulation time step in seconds.
gravity: 3-vector indicating the gravity feild in world frame.
 WARNING: it CANNOT be changed after the simulator object is constructed.
 TODO: differentiate gravity from other simulation parameters, which are not
  ignored when calling QuasistaticSimulator.Step(...).
nd_per_contact: int, number of extreme rays per contact point. Only
 useful in QP mode.

contact_detection_tolerance: Signed distance pairs whose distances are
 greater than this value are ignored in the simulator's non-penetration
 constraints. Unit is in meters.

is_quasi_dynamic: bool. If True, dynamics of unactauted objects is
 given by sum(F) = M @ (v_(l+1) - 0). If False, it becomes sum(F) = 0 instead.

 The mass matrix for unactuated objects is always added when the
 unconstrained (log-barrier) version of the problem is solved. Not having a mass
 matrix can sometimes make the unconstrained program unbounded.

mode:
Note that C++ does not support modes using CVX.
                 | Friction Cone | Force Field | Parser   |
kQpMp            | Pyramid       | No          | MP       |
kQpCvx           | Pyramid       | No          | CVXPY    |
kSocpMp          | Icecream      | No          | MP       |
kLogPyramidMp    | Pyramid       | Yes         | MP       |
kLogPyramidMy    | Pyramid       | Yes         | in-house |
kLogPyramidCvx   | Pyramid       | Yes         | CVXPY    |
kLogIcecream     | Icecream      | Yes         | in-house |

log_barrier_weight: float, used only in log-barrier modes.

unactuated_mass_scale:
scales the mass matrix of un-actuated objects by epsilon, so that
(max_M_u_eigen_value * epsilon) * unactuated_mass_scale = min_h_squared_K.
    If 0, the mass matrix is not scaled. Refer to the function that computes
    mass matrix for details.
*------------------------------C++ only-----------------------------------*
gradient_lstsq_tolerance: float
   When solving for A during dynamics gradient computation, i.e.
   A * A_inv = I_n, --------(*)
   the relative error is defined as
   (A_sol * A_inv - I_n) / n,
   where A_sol is the least squares solution to (*), or the pseudo-inverse
   of A_inv.
   A warning is printed when the relative error is greater than this number.
*/
// TODO: the inputs to QuasistaticSimulator's constructor should be
//  collected into a "QuasistaticPlantParameters" structure, which
//  cannot be changed after the constructor call. "gravity" belongs there.
struct QuasistaticSimParameters {
  double h{NAN};
  Eigen::Vector3d gravity{0, 0, 0};
  double contact_detection_tolerance{NAN};
  bool is_quasi_dynamic{true};
  double log_barrier_weight{NAN};
  double unactuated_mass_scale{NAN};
  /*
   * Some computation can be saved by setting this to false.
   * Contact forces are only computed for "exact" forward dynamics, i.e.
   * kQpMp and kSocpMp.
   */
  bool calc_contact_forces{true};
  // -------------------------- CPP only --------------------------
  double gradient_lstsq_tolerance{0.3};
  // -------------------------- Not Set in YAML -------------------------
  ForwardDynamicsMode forward_mode{ForwardDynamicsMode::kQpMp};
  GradientMode gradient_mode{GradientMode::kNone};
  // ---------------------- pyramid cones only ---------------------------
  size_t nd_per_contact{0};
  // free solvers: SCS for cone programs, OSQP for QPs.
  bool use_free_solvers{false};
};

static char const *const kMultiBodyPlantName = "MultiBodyPlant";
static char const *const kSceneGraphName = "SceneGraph";
