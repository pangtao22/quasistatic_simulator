import os
import pathlib
from typing import List

import numpy as np
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.math import RigidTransform

from contact_aware_control.plan_runner.setup_three_link_arm import (
    robot_sdf_path, ground_sdf_path)

module_path = pathlib.Path(__file__).parent.absolute()
# transform between robot base frame and world frame
X_WR = RigidTransform()
X_WR.set_translation([0, 0, 0.1])


def Create3LinkArmControllerPlant():
    # creates plant that includes only the robot, used for controllers.
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)
    parser.AddModelFromFile(robot_sdf_path)
    # plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("link_0"), X_WR)
    plant.Finalize()
    return plant


def CreatePlantFor2dArmWithMultipleObjects(builder,
                                           object_sdf_paths: List[str]):
    """
    :param builder: a DiagramBuilder object.
    :param object_sdf_paths: list of absolute paths to object.sdf files.
    :return:
    """
    # MultibodyPlant
    plant = MultibodyPlant(1e-3)
    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)

    # Add ground
    parser.AddModelFromFile(ground_sdf_path)
    X_WG = RigidTransform.Identity()
    X_WG.set_translation([0, 0, -0.5])  # "ground"
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("ground"),
                     X_AB=X_WG)

    # Add Robot
    robot_model = parser.AddModelFromFile(robot_sdf_path)
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("link_0"), X_WR)
    # plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])

    # Add objects
    object_models_list = []
    for i, sdf_path in enumerate(object_sdf_paths):
        object_models_list.append(
            parser.AddModelFromFile(sdf_path, model_name="box%d" % i))

    plant.Finalize()

    return (plant,
            scene_graph,
            [robot_model],
            object_models_list)


def CreatePlantFor2dGripper(builder, *args):
    """
    This function should be called when constructing a Diagram in RobotSimulator.
    :param builder: a reference to the DiagramBuilder.
    :return:
    """
    # MultibodyPlant
    plant = MultibodyPlant(1e-3)
    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])

    # Add ground
    parser.AddModelFromFile(ground_sdf_path)
    X_WG = RigidTransform.Identity()
    X_WG.set_translation([0, 0, -0.5])  # "ground"
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("ground"),
                     X_AB=X_WG)

    # Add robot.
    gripper_sdf_path = os.path.join(module_path, "models", "gripper.sdf")
    robot_model = parser.AddModelFromFile(gripper_sdf_path)

    # Add object
    object_sdf_path = os.path.join(module_path, "models", "sphere_yz.sdf")
    object_model = parser.AddModelFromFile(object_sdf_path)

    plant.Finalize()

    return (plant,
            scene_graph,
            [robot_model],
            [object_model])