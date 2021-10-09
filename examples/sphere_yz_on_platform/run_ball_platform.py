import os
import numpy as np

from pydrake.all import RigidTransform, DiagramBuilder, PiecewisePolynomial

from examples.setup_simulation_diagram import (
    run_quasistatic_sim, shift_q_traj_to_start_at_minus_h)
from core.quasistatic_simulator import (
    QuasistaticSimParameters, create_plant_with_robots_and_objects)
from examples.model_paths import models_dir

model_directive_path = os.path.join(models_dir,
                                    "ball_and_platform.yml")

#%% sim setup
h = 0.05
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h
quasistatic_sim_params = QuasistaticSimParameters(
    gravity=np.array([0, 0, -10.]),
    nd_per_contact=2,
    contact_detection_tolerance=np.inf,
    is_quasi_dynamic=True,
    requires_grad=True)

# robot
Kp = np.array([100, 100], dtype=float)
robot_name = "hand"
robot_stiffness_dict = {robot_name: Kp}


# trajectory and initial conditions.
nq_a = 2
qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [-0.6, 1.2]
qa_knots[1] = [-0.6, 1.2]
qa_traj = PiecewisePolynomial.ZeroOrderHold([0, duration], qa_knots.T)
q_a_traj_dict_str = {robot_name: qa_traj}
q0_dict_str = {robot_name: qa_knots[0]}


#%% run sim.
if __name__ == "__main__":
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        model_directive_path=model_directive_path,
        object_sdf_paths={},
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict={robot_name: Kp},
        h=h,
        sim_params=quasistatic_sim_params,
        is_visualizing=True,
        real_time_rate=1.0)
