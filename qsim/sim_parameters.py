import enum
import sys
from collections import namedtuple
from qsim_cpp import GradientMode, QuasistaticSimParametersCpp
import numpy as np

"""
:param nd_per_contact: int, number of extreme rays per contact point.
:param contact_detection_tolerance: Signed distance pairs whose distances are 
    greater than this value are ignored in the simulator's non-penetration 
    constraints. Unit is in meters.
:param is_quasi_dynamic: bool. If True, dynamics of unactauted objects is 
    given by sum(F) = M @ (v_(l+1) - 0). If False, it becomes sum(F) = 0 
    instead. 
    The mass matrix for unactuated objects is always added when the 
    unconstrained version of the problem is solved. Not having a mass 
    matrix can sometimes makes the unconstrained program unbounded. 
/*----------------------------------------------------------------------*/
/*---------Experimental features only supported in python.--------------*/
:param mode: Union['qp_mp', 'qp_cvx', 'unconstrained']. 
    - 'qp_mp': solves the standard QP for system states at the next time 
        step, using MathematicalProgram. 
    - 'qp_mp': solves the standard QP using cvxpy.
    - 'log_cvx': solves an unconstrained version of the QP, obtained by 
        moving inequality constraints into the objective with 
        log barrier functions.
    - 'log_mp': same problem formulation as log_cvx, but uses 
        MathematicalProgram.
:param log_barrier_weight: float, used only when is_unconstrained == True.

/*----------------------------------------------------------------------*/
:param requires_grad: whether the gradient of v_next w.r.t the parameters of 
    the QP are computed. 
    Note that this parameter is only effective in 
        QuasistaticSimulator.step_default(...),
    which is only used by QuasistaticSystem, which almost never computes 
    gradients. In applications that does compute gradient, such as 
    QuasistaticDynamics from irs_lqr, QuasistaticSimulator.step function is 
    invoked and a separate GradientMode value is passed to it explicitly. 
:param gradient_from_active_constraints: bool. Whether the dynamics gradient is 
    computed from all constraints or only the active constraints.
    
/*----------------------------------------------------------------------*/
unactuated_mass_scale: 
scales the mass matrix of un-actuated objects by epsilon, so that 
(max_M_u_eigen_value * epsilon) * unactuated_mass_scale = min_h_squared_K.
If 0, the mass matrix is not scaled.
"""
field_names = [
    "gravity", "nd_per_contact", "contact_detection_tolerance",
    "is_quasi_dynamic", "mode", "log_barrier_weight", "gradient_mode",
    "grad_from_active_constraints", "unactuated_mass_scale"
]
defaults = [np.array([0, 0, -9.81]), 4, 0.01,
            False, "qp_mp", 1e4, GradientMode.kNone, True, np.nan]

if sys.version_info >= (3, 7):
    QuasistaticSimParameters = namedtuple(
        "QuasistaticSimParameters",
        field_names=field_names,
        defaults=defaults)
else:
    QuasistaticSimParameters = namedtuple(
        "QuasistaticSimParameters",
        field_names=field_names)
    QuasistaticSimParameters.__new__.__defaults__ = tuple(defaults)
    QuasistaticSimParameters = QuasistaticSimParameters


def cpp_params_from_py_params(
        sim_params: QuasistaticSimParameters) -> QuasistaticSimParametersCpp:
    sim_params_cpp = QuasistaticSimParametersCpp()
    sim_params_cpp.gravity = sim_params.gravity
    sim_params_cpp.nd_per_contact = sim_params.nd_per_contact
    sim_params_cpp.contact_detection_tolerance = (
        sim_params.contact_detection_tolerance)
    sim_params_cpp.is_quasi_dynamic = sim_params.is_quasi_dynamic
    sim_params_cpp.gradient_mode = sim_params.gradient_mode
    sim_params_cpp.gradient_from_active_constraints = (
        sim_params.grad_from_active_constraints)
    sim_params_cpp.unactuated_mass_scale = sim_params.unactuated_mass_scale
    return sim_params_cpp
