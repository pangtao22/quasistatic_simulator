from typing import List, Dict

import numpy as np
from pydrake.all import (
    DiagramBuilder,
    DiscreteContactSolver,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    PdControllerGains,
)

from .model_paths import add_package_paths_local


def create_plant_with_robots_and_objects(
    builder: DiagramBuilder,
    model_directive_path: str,
    robot_names: List[str],
    object_sdf_paths: Dict[str, str],
    time_step: float,
    gravity: np.ndarray,
    mbp_solver: DiscreteContactSolver,
    add_robot_pd_controller: bool = False,
    robot_stiffness_dict: Dict[str, np.ndarray] = None,
    robot_damping_dict: Dict[str, np.ndarray] = None,
):
    """
    Add plant and scene_graph constructed from a model_directive to builder.
    """

    # MultibodyPlant
    plant = MultibodyPlant(time_step)
    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    add_package_paths_local(parser)

    # Objects
    # It is important that object_models and robot_models are ordered.
    object_models = set()
    for name, sdf_path in object_sdf_paths.items():
        object_models.add(parser.AddModelFromFile(sdf_path, model_name=name))

    # Robots
    ProcessModelDirectives(
        LoadModelDirectives(model_directive_path), plant, parser
    )
    robot_models = set()
    for name in robot_names:
        robot_model = plant.GetModelInstanceByName(name)
        robot_models.add(robot_model)

    # gravity
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    # Discrete-time Solver.
    plant.set_discrete_contact_solver(mbp_solver)

    if add_robot_pd_controller:
        add_implicit_pd_controller(
            plant=plant,
            robot_stiffness_dict=robot_stiffness_dict,
            robot_damping_dict=robot_damping_dict,
        )

    plant.Finalize()

    return plant, scene_graph, robot_models, object_models


def add_implicit_pd_controller(
    plant: MultibodyPlant,
    robot_stiffness_dict: Dict[str, np.ndarray],
    robot_damping_dict: Dict[str, np.ndarray],
):
    for robot_name, joint_stiffness in robot_stiffness_dict.items():
        robot_model = plant.GetModelInstanceByName(robot_name)
        joint_damping = robot_damping_dict[robot_name]
        assert len(joint_stiffness) == len(joint_damping)

        joint_indices = plant.GetJointIndices(robot_model)
        actuated_joint_indices = []
        for joint_index in joint_indices:
            if not plant.HasJointActuatorNamed(
                plant.get_joint(joint_index).name()
            ):
                continue
            actuated_joint_indices.append(joint_index)
        assert len(actuated_joint_indices) == len(joint_stiffness)

        for i, joint_index in enumerate(actuated_joint_indices):
            joint = plant.get_joint(joint_index)
            actuator = plant.GetJointActuatorByName(joint.name())
            actuator.set_controller_gains(
                PdControllerGains(p=joint_stiffness[i], d=joint_damping[i])
            )
