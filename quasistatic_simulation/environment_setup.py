from collections import namedtuple
from typing import Dict

import numpy as np
from pydrake.all import (MultibodyPlant, Parser, RigidTransform,
                         DiagramBuilder, AddMultibodyPlantSceneGraph)

from contact_aware_control.plan_runner.setup_three_link_arm import (
    ground_sdf_path)

RobotInfo = namedtuple("RobotInfo", ["sdf_path", "parent_model_name",
                                     "parent_frame_name", "base_frame_name",
                                     "X_PB", "joint_stiffness"])


def add_ground_to_plant(plant: MultibodyPlant, parser: Parser):
    parser.AddModelFromFile(ground_sdf_path)
    X_WG = RigidTransform.Identity()
    X_WG.set_translation([0, 0, -0.5])  # "ground"
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("ground"),
                     X_AB=X_WG)


def create_plant_with_robots_and_objects(
        builder: DiagramBuilder,
        robot_info_dict: Dict[str, RobotInfo],
        object_sdf_paths: Dict[str, str],
        time_step: float,
        gravity: np.ndarray):

    # MultibodyPlant
    plant = MultibodyPlant(time_step)
    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)

    # ground
    add_ground_to_plant(plant, parser)

    # Robots
    robot_models_list = []
    for name, robot_info in robot_info_dict.items():
        robot_model = parser.AddModelFromFile(robot_info.sdf_path, name)
        robot_models_list.append(robot_model)

        if robot_info.parent_model_name is None:
            continue
        parent_model = plant.GetModelInstanceByName(
            robot_info.parent_model_name)
        parent_frame = plant.GetFrameByName(
            robot_info.parent_frame_name, parent_model)
        robot_base_frame = plant.GetFrameByName(
            robot_info.base_frame_name, robot_model)
        plant.WeldFrames(A=parent_frame,
                         B=robot_base_frame,
                         X_AB=robot_info.X_PB)


    # Objects
    object_models_list = []
    for name, sdf_path in object_sdf_paths.items():
        object_models_list.append(
            parser.AddModelFromFile(sdf_path, model_name=name))

    # gravity
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    plant.Finalize()

    return plant, scene_graph, robot_models_list, object_models_list
