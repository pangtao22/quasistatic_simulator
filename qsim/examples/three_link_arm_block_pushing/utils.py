import os.path
from typing import Dict

import numpy as np
from qsim.examples.setup_simulations import run_quasistatic_sim, run_mbp_sim
from pydrake.all import (
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    PiecewisePolynomial,
    MultibodyPlant,
)
from qsim.model_paths import add_package_paths_local, models_dir
from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from robotics_utilities.iiwa_controller.robot_internal_controller import (
    RobotInternalController,
)

from qsim.utils import is_mosek_gurobi_available


# Simulation parameters.
q_model_path_2d = "q_sys/3_link_arm_2d_box.yml"
q_model_path_3d = "q_sys/3_link_arm_3d_box.yml"
robot_name = "arm"
box_name = "box0"
h_quasistatic = 0.02
h_mbp = 1e-3

# Robot joint trajectory.
nq_a = 3
qa_knots = np.zeros((2, nq_a))
qa_knots[0] = [np.pi / 2, -np.pi / 2, -np.pi / 2]
qa_knots[1] = qa_knots[0] + 0.3
q_robot_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    breaks=[0, 10],
    samples=qa_knots.T,
    sample_dot_at_start=np.zeros(nq_a),
    sample_dot_at_end=np.zeros(nq_a),
)


def create_3link_arm_controller_plant(gravity: np.ndarray):
    # creates plant that includes only the robot, used for controllers.
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)
    add_package_paths_local(parser)
    controller_model_directive = os.path.join(models_dir, "three_link_arm.yml")
    ProcessModelDirectives(
        LoadModelDirectives(controller_model_directive), plant, parser
    )
    plant.mutable_gravity_field().set_gravity_vector(gravity)
    plant.Finalize()
    return plant, None


def run_mbp_quasistatic_comparison(
    quasistatic_model_path: str,
    q0_dict_str: Dict[str, np.ndarray],
    is_visualizing=False,
    real_time_rate=0.0,
):
    q_parser = QuasistaticParser(
        os.path.join(models_dir, quasistatic_model_path)
    )
    q_parser.set_sim_params(
        h=h_quasistatic,
        use_free_solvers=not is_mosek_gurobi_available(),
    )

    # Quasistatic
    loggers_dict_quasistatic_str, q_sys = run_quasistatic_sim(
        q_parser=q_parser,
        backend=QuasistaticSystemBackend.PYTHON,
        q_a_traj_dict_str={robot_name: q_robot_traj},
        q0_dict_str=q0_dict_str,
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate,
    )

    # MBP
    # create controller system for robot.
    plant_robot, _ = create_3link_arm_controller_plant(q_parser.get_gravity())
    controller_robot = RobotInternalController(
        plant_robot=plant_robot,
        joint_stiffness=q_parser.get_robot_stiffness_by_name(robot_name),
        controller_mode="impedance",
    )
    robot_controller_dict = {robot_name: controller_robot}

    loggers_dict_mbp_str = run_mbp_sim(
        model_directive_path=q_parser.model_directive_path,
        object_sdf_paths=q_parser.object_sdf_paths,
        q_a_traj_dict={robot_name: q_robot_traj},
        q0_dict_str=q0_dict_str,
        robot_stiffness_dict=q_parser.robot_stiffness_dict,
        robot_controller_dict=robot_controller_dict,
        h=h_mbp,
        gravity=q_parser.get_gravity(),
        is_visualizing=is_visualizing,
        real_time_rate=real_time_rate,
    )

    # Extracting iiwa configuration logs.
    n_qu = q0_dict_str[box_name].size
    q_box_log_mbp = loggers_dict_mbp_str[box_name].data()[:n_qu].T
    q_robot_log_mbp = loggers_dict_mbp_str[robot_name].data()[:nq_a].T
    t_mbp = loggers_dict_mbp_str[robot_name].sample_times()

    q_robot_log_quasistatic = loggers_dict_quasistatic_str[robot_name].data().T
    q_box_log_quasistatic = loggers_dict_quasistatic_str[box_name].data().T
    t_quasistatic = loggers_dict_quasistatic_str[robot_name].sample_times()

    return (
        q_robot_log_mbp,
        q_box_log_mbp,
        t_mbp,
        q_robot_log_quasistatic,
        q_box_log_quasistatic,
        t_quasistatic,
        q_sys,
    )
