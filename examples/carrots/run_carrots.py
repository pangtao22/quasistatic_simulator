import os
import numpy as np

from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import (
    run_quasistatic_sim)
from qsim.simulator import QuasistaticSimParameters
from qsim.model_paths import models_dir

object_sdf_path = os.path.join(models_dir, "sphere_yz_small.sdf")
model_directive_path = os.path.join(models_dir, "carrot.yml")

#%% sim setup
h = 0.2
T = int(round(3 / h))  # num of time steps to simulate forward.
duration = T * h
sim_settings = QuasistaticSimParameters(
    gravity=np.array([0, 0, 0]),
    nd_per_contact=2,
    contact_detection_tolerance=0.1,
    is_quasi_dynamic=True)

# robots.
Kp = np.array([50, 50, 100, 50, 50], dtype=float)
robot_stiffness_dict = {"gripper": Kp}

# object
num_pieces = 100
object_sdf_dict = {
    "carrot_{:02d}".format(i): object_sdf_path for i in range(num_pieces)}

# trajectory and initial conditions.
nq_a = 5
qa_knots = np.zeros((2, nq_a))
# x, y, z, dy1, dy2
qa_knots[0] = [-0.1, 0.25, 0, -0.05, 0.05]
qa_knots[1] = [0.4, 0.25, 0, -0.05, 0.05]
q_robot_traj = PiecewisePolynomial.FirstOrderHold(
    [0, h * T], qa_knots.T)

q_a_traj_dict_str = {"gripper": q_robot_traj}

q0_dict_str = {"gripper": qa_knots[0]}

q_u0 = 0.5 * np.random.rand(num_pieces,2)
for i in range(num_pieces):
    object_name = "carrot_{:02d}".format(i)
    q0_dict_str[object_name] = q_u0[i]

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
