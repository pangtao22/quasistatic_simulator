import enum
import sys
from collections import namedtuple
from qsim_cpp import GradientMode
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
    - 'unconstrained': solves an unconstrained version of the QP, obtained by 
        moving inequality constraints into the objective with 
        log barrier functions. 
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
"""
field_names = [
    "gravity", "nd_per_contact", "contact_detection_tolerance",
    "is_quasi_dynamic", "mode", "log_barrier_weight", "gradient_mode",
    "grad_from_active_constraints"
]
defaults = [np.array([0, 0, -9.81]), 4, 0.01,
            False, "qp_mp", 1e4, GradientMode.kAB, True]

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
