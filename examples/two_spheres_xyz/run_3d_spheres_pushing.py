import os

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import PiecewisePolynomial

from examples.setup_simulations import run_quasistatic_sim
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.simulator import ForwardDynamicsMode, GradientMode
from qsim.model_paths import models_dir



#%% sim setup
q_model_path = os.path.join(models_dir, 'q_sys', 'two_spheres_xyz.yml')

h = 0.1
T = int(round(2 / h))  # num of time steps to simulate forward.
duration = T * h

# robot
robot_name = "sphere_xyz_actuated"
object_name = "sphere_xyz"
r_robot = 0.1
r_obj = 0.5


# trajectory and initial conditions.
nq_a = 3
theta = np.pi / 12

qa_knots = np.zeros((3, nq_a))
qa_knots[0] = [-np.cos(theta) * (r_robot + r_obj),
               -np.sin(theta) * (r_robot + r_obj),
               r_obj]
qa_knots[1] = [np.cos(theta) * 1.0,
               np.sin(theta) * 1.0,
               r_obj]
qa_knots[2] = qa_knots[1]
qa_traj = PiecewisePolynomial.FirstOrderHold([0, duration * 0.8, duration],
                                             qa_knots.T)
q_a_traj_dict_str = {robot_name: qa_traj}
qu0 = np.array([0., 0., r_obj])
q0_dict_str = {object_name: qu0,
               robot_name: qa_knots[0]}


q_parser = QuasistaticParser(q_model_path)
q_parser.set_sim_params(
    h=h,
    is_quasi_dynamic=True,
    forward_mode=ForwardDynamicsMode.kSocpMp,
    log_barrier_weight=100)


loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
    q_parser=q_parser,
    backend=QuasistaticSystemBackend.CPP,
    q_a_traj_dict_str=q_a_traj_dict_str,
    q0_dict_str=q0_dict_str,
    is_visualizing=True,
    real_time_rate=0.1)


#%%
q_obj_trj = loggers_dict_quasistatic_str[object_name].data()
q_robot_trj = loggers_dict_quasistatic_str[robot_name].data()
t = loggers_dict_quasistatic_str[object_name].sample_times()
q_robot_cmd = np.array([qa_traj.value(t_i).squeeze() for t_i in t]).T

# compute angles.
T = len(t)
angles = np.zeros(T)
for i in range(T - 1):
    d_xy = q_obj_trj[:2, i + 1] - q_obj_trj[:2, i]
    angles[i] = np.arctan2(d_xy[1], d_xy[0])


#%%
fig, axes = plt.subplots(1, 3)
for i, ax in enumerate(axes):
    ax.plot(t, q_obj_trj[i])
    ax.grid(True)
plt.show()


plt.figure()
plt.plot(q_obj_trj[0], q_obj_trj[1])
plt.plot(q_robot_trj[0], q_robot_trj[1], '--')
plt.plot(q_robot_cmd[0], q_robot_cmd[1], '--')
plt.axis('equal')
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(angles)
plt.axhline(theta)
plt.grid()
plt.show()


