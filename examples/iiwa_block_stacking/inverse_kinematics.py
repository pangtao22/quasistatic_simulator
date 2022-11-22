import numpy as np

from pydrake.all import (
    InverseKinematics,
    RotationMatrix,
    MultibodyPlant,
    PiecewiseQuaternionSlerp,
    PiecewisePolynomial,
)
from pydrake.solvers import mathematicalprogram as mp


def calc_iwa_trajectory_for_point_tracking(
    plant: MultibodyPlant,
    duration: float,
    num_knot_points: int,
    p_WQ_start: np.ndarray,
    p_WQ_offset: np.ndarray,
    R_WL7_start: RotationMatrix,
    R_WL7_final: RotationMatrix,
    q_initial_guess: np.ndarray,
    p_L7Q: np.ndarray,
):
    """
    Solves for a joint angle trajector for IIWA such that point Q,
    fixed relative to frame L7, follows the straight line from p_WQ_start to
    (p_WQ_start + p_WQ_offset). The orientation of frame L7 is interpolated
    lienarly from R_WL7_start to R_WL7_final.
    """
    theta_bound = 0.001
    position_tolerance = 0.005
    l7_frame = plant.GetBodyByName("iiwa_link_7").body_frame()

    def InterpolatePosition(i):
        return p_WQ_start + p_WQ_offset / (num_knot_points - 1) * i

    q_knots = np.zeros((num_knot_points, plant.num_positions()))
    t_knots = np.linspace(0, duration, num_knot_points)
    R_WL7_traj = PiecewiseQuaternionSlerp(
        [0, duration], [R_WL7_start.ToQuaternion(), R_WL7_final.ToQuaternion()]
    )

    for i in range(0, num_knot_points):
        ik = InverseKinematics(plant)
        q_variables = ik.q()
        R_WL7_r = RotationMatrix(R_WL7_traj.orientation(t_knots[i]))

        ik.AddOrientationConstraint(
            frameAbar=plant.world_frame(),
            R_AbarA=R_WL7_r,
            frameBbar=l7_frame,
            R_BbarB=RotationMatrix.Identity(),
            theta_bound=theta_bound,
        )

        # Position constraint
        p_WQ = InterpolatePosition(i)
        ik.AddPositionConstraint(
            frameB=l7_frame,
            p_BQ=p_L7Q,
            frameA=plant.world_frame(),
            p_AQ_lower=p_WQ - position_tolerance,
            p_AQ_upper=p_WQ + position_tolerance,
        )

        prog = ik.prog()
        # use the robot posture at the previous knot point as
        # an initial guess.
        if i == 0:
            prog.SetInitialGuess(q_variables, q_initial_guess)
        else:
            prog.SetInitialGuess(q_variables, q_knots[i - 1])
        result = mp.Solve(prog)
        print(i, ": ", result.get_solution_result())
        q_knots[i] = result.GetSolution(q_variables)

    q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        t_knots, q_knots.T, np.zeros(7), np.zeros(7)
    )

    return q_traj, q_knots
