import os.path
from typing import Dict

import numpy as np
from qsim.simulator import (
    QuasistaticSimParameters)
from examples.model_paths import add_package_paths_local, models_dir
from examples.setup_simulation_diagram import run_quasistatic_sim, run_mbp_sim
from pydrake.all import (Parser, ProcessModelDirectives, LoadModelDirectives,
                         PiecewisePolynomial, MultibodyPlant)
from robotics_utilities.iiwa_controller.robot_internal_controller import (
    RobotInternalController)

# Simulation parameters.
gravity = np.array([0, 0, -10.])
robot_name = "arm"
box_name = "box0"
Kp = np.array([1000, 1000, 1000], dtype=float)
robot_stiffness_dict = {robot_name: Kp}
h_quasistatic = 0.02
h_mbp = 1e-3

# Robot joint trajectory.
nq_a = 3
qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [np.pi / 2, -np.pi / 2, -np.pi / 2]
qa_knots[1] = qa_knots[0] + 0.3
q_robot_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=[0, 10], samples=qa_knots.T,
    sample_dot_at_start=np.zeros(nq_a),
    sample_dot_at_end=np.zeros(nq_a))

# model directive paths
model_directive_path = os.path.join(
    models_dir, 'three_link_arm_and_ground.yml')


def create_3link_arm_controller_plant(gravity: np.ndarray):
    # creates plant that includes only the robot, used for controllers.
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)
    add_package_paths_local(parser)
    controller_model_directive = os.path.join(
        models_dir, 'three_link_arm.yml')
    ProcessModelDirectives(LoadModelDirectives(controller_model_directive),
                           plant, parser)
    plant.mutable_gravity_field().set_gravity_vector(gravity)
    plant.Finalize()
    return plant, None


def run_mbp_quasistatic_comparison(box_sdf_path: str,
                                   q0_dict_str: Dict[str, np.ndarray],
                                   quasistatic_sim_params: QuasistaticSimParameters,
                                   is_visualizing=False, real_time_rate=0.0):
    # Quasistatic
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        model_directive_path=model_directive_path,
        object_sdf_paths={box_name: box_sdf_path},
        q_a_traj_dict_str={robot_name: q_robot_traj},
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict=robot_stiffness_dict,
        h=h_quasistatic,
        sim_params=quasistatic_sim_params,
        is_visualizing=is_visualizing, real_time_rate=real_time_rate)

    # MBP
    # create controller system for robot.
    plant_robot, _ = create_3link_arm_controller_plant(gravity)
    controller_robot = RobotInternalController(
        plant_robot=plant_robot, joint_stiffness=Kp,
        controller_mode="impedance")
    robot_controller_dict = {robot_name: controller_robot}

    loggers_dict_mbp_str = run_mbp_sim(
        model_directive_path=model_directive_path,
        object_sdf_paths={box_name: box_sdf_path},
        q_a_traj_dict={robot_name: q_robot_traj},
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict=robot_stiffness_dict,
        robot_controller_dict=robot_controller_dict,
        h=h_mbp,
        gravity=gravity,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate)

    # Extracting iiwa configuration logs.
    n_qu = q0_dict_str[box_name].size
    q_box_log_mbp = loggers_dict_mbp_str[box_name].data()[:n_qu].T
    q_robot_log_mbp = loggers_dict_mbp_str[robot_name].data()[:nq_a].T
    t_mbp = loggers_dict_mbp_str[robot_name].sample_times()

    q_robot_log_quasistatic = loggers_dict_quasistatic_str[robot_name].data().T
    q_box_log_quasistatic = loggers_dict_quasistatic_str[box_name].data().T
    t_quasistatic = loggers_dict_quasistatic_str[robot_name].sample_times()

    return (q_robot_log_mbp, q_box_log_mbp, t_mbp,
            q_robot_log_quasistatic, q_box_log_quasistatic, t_quasistatic,
            q_sys)
