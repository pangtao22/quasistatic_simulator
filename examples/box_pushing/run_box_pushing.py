import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulation_diagram import (
    run_quasistatic_sim)
from qsim.simulator import QuasistaticSimParameters
from qsim.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "box_1m_rotation.sdf")
model_directive_path = os.path.join(models_dir, "box_pushing.yml")

#%% sim setup
h = 0.2
T = int(round(5. / h))  # num of time steps to simulate forward.
duration = T * h
print(duration)
sim_settings = QuasistaticSimParameters(
    gravity=np.array([0, 0, 0.]),
    nd_per_contact=2,
    contact_detection_tolerance=0.1,
    is_quasi_dynamic=True)

# robots.
Kp = np.array([50, 50], dtype=float)
robot_stiffness_dict = {"hand": Kp}

# object
object_name = "box"
object_sdf_dict = {object_name: object_sdf_path}

# trajectory and initial conditions.
nq_a = 2
qa_knots = np.zeros((2, nq_a))
# x, y, z, dy1, dy2
qa_knots[0] = [0.0, -0.2]
qa_knots[1] = [0.1, 1.0]
q_robot_traj = PiecewisePolynomial.FirstOrderHold(
    [0, T * h], qa_knots.T)

q_a_traj_dict_str = {"hand": q_robot_traj}

q_u0 = np.array([0, 0.5, 0])

q0_dict_str = {"box": q_u0,
               "hand": qa_knots[0]}


#%% run sim.
if __name__ == "__main__":
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        model_directive_path=model_directive_path,
        object_sdf_paths=object_sdf_dict,
        q_a_traj_dict_str=q_a_traj_dict_str,
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict=robot_stiffness_dict,
        h=h,
        sim_params=sim_settings,
        is_visualizing=True,
        real_time_rate=1.0)
