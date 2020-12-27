from pydrake.all import (PiecewisePolynomial, TrajectorySource, Simulator)

from contact_aware_control.plan_runner.plan_utils import (
    RenderSystemWithGraphviz)

from quasistatic_simulation.quasistatic_system import *


def run_sim_quasistatic(
        q_iiwa_traj: PiecewisePolynomial,
        q_schunk_traj: PiecewisePolynomial,
        q_u0_list: np.array,
        Kp_iiwa: np.array,
        Kp_schunk: np.array,
        object_sdf_paths: List[str],
        setup_environment: SetupEnvironmentFunction,
        time_step: float):

    pass

