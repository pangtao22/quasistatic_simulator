import os
import pathlib
from typing import List, Tuple, Callable

import numpy as np

from pydrake.all import (ModelInstanceIndex, DiagramBuilder, SceneGraph,
                         Parser, MultibodyPlant, AddMultibodyPlantSceneGraph,
                         FrameIndex)
from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from contact_aware_control.plan_runner.setup_three_link_arm import (
    robot_sdf_path, ground_sdf_path)
from contact_aware_control.plan_runner.setup_iiwa import (
    iiwa_sdf_path_drake, ee_sdf_path)

"""
Input:
    np.array: (3,) gravity vector.
    
:returns
    MultibodyPlant: the MBP containing the robot and objects.
    List[FrameIndex]: a list containing all indices of all body frames of the 
        robot.
"""
CreateControllerPlantFunction = Callable[
    [np.array], Tuple[MultibodyPlant, List[FrameIndex]]]


schunk_sdf_path = FindResourceOrThrow(
    "drake/manipulation/models/wsg_50_description/sdf"
    "/schunk_wsg_50_ball_contact.sdf")

# transform between robot base frame and world frame
X_WR = RigidTransform()
X_WR.set_translation([0, 0, 0.1])

model_dir_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(), "..", "models")
box2d_big_sdf_path = os.path.join(model_dir_path, "box_yz_rotation_big.sdf")
box3d_big_sdf_path = os.path.join(model_dir_path, "box_1m.sdf")
box3d_medium_sdf_path = os.path.join(model_dir_path, "box_0.6m.sdf")
box3d_small_sdf_path = os.path.join(model_dir_path, "box_0.5m.sdf")
box3d_8cm_sdf_path = os.path.join(model_dir_path, "box_0.08m.sdf")
box3d_7cm_sdf_path = os.path.join(model_dir_path, "box_0.07m.sdf")
box3d_6cm_sdf_path = os.path.join(model_dir_path, "box_0.06m.sdf")


def create_3link_arm_controller_plant(gravity: np.ndarray):
    # creates plant that includes only the robot, used for controllers.
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)
    parser.AddModelFromFile(robot_sdf_path)
    plant.mutable_gravity_field().set_gravity_vector(gravity)
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("link_0"), X_WR)
    plant.Finalize()
    return plant, None


def create_3link_arm_plant_with_multiple_objects(
        builder, object_sdf_paths: List[str], time_step: float,
        gravity: np.ndarray):
    """
    :param builder: a DiagramBuilder object.
    :param object_sdf_paths: list of absolute paths to object.sdf files.
    :return:
    """
    # MultibodyPlant
    plant = MultibodyPlant(time_step)
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

    # Add objects
    object_models_list = []
    for i, sdf_path in enumerate(object_sdf_paths):
        object_models_list.append(
            parser.AddModelFromFile(sdf_path, model_name="box%d" % i))

    # gravity
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    plant.Finalize()

    return (plant,
            scene_graph,
            [robot_model],
            object_models_list)


def create_iiwa_plant_with_multiple_objects(builder,
                                            object_sdf_paths: List[str]):
    """
    A SetupEnvironmentFunction.
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

    # fix robot to world
    robot_model = parser.AddModelFromFile(iiwa_sdf_path_drake)
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("iiwa_link_0"),
                     X_AB=RigidTransform.Identity())

    # fix ee_sphere to l7
    ee_model = parser.AddModelFromFile(ee_sdf_path)
    X_L7E = RigidTransform()
    X_L7E.set_translation([0, 0, 0.075])
    plant.WeldFrames(A=plant.GetFrameByName("iiwa_link_7"),
                     B=plant.GetFrameByName("body", ee_model),
                     X_AB=X_L7E)

    plant.mutable_gravity_field().set_gravity_vector(gravity)

    # Add objects
    object_models_list = []
    for i, sdf_path in enumerate(object_sdf_paths):
        object_models_list.append(
            parser.AddModelFromFile(sdf_path, model_name="box%d" % i))

    plant.Finalize()

    return (plant,
            scene_graph,
            [robot_model, ee_model],
            object_models_list)


def create_iiwa_plant(builder, object_sdf_paths: List[str],
                      time_step: float, gravity: np.ndarray):
    # MultibodyPlant
    plant = MultibodyPlant(time_step)
    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)

    # fix robot to world
    iiwa_model = parser.AddModelFromFile(iiwa_sdf_path_drake)
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("iiwa_link_0"),
                     X_AB=RigidTransform.Identity())

    plant.mutable_gravity_field().set_gravity_vector(gravity)
    plant.Finalize()

    return plant, scene_graph, [iiwa_model], []


def create_iiwa_plant_with_schunk(
        builder, object_sdf_paths: List[str], time_step: float,
        gravity: np.ndarray):
    """
    :param builder: a DiagramBuilder object.
    :param object_sdf_paths: list of absolute paths to object.sdf files.
    :return:
    """
    # MultibodyPlant
    plant = MultibodyPlant(time_step)
    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)

    # Add ground
    parser.AddModelFromFile(ground_sdf_path)
    X_WG = RigidTransform.Identity()
    X_WG.set_translation([0, 0, -0.5])  # "ground"
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("ground"),
                     X_AB=X_WG)

    # fix robot to world
    robot_model = parser.AddModelFromFile(iiwa_sdf_path_drake)
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("iiwa_link_0"),
                     X_AB=RigidTransform.Identity())

    # fix schunk to l7
    schunk_model = parser.AddModelFromFile(schunk_sdf_path)
    X_L7E = RigidTransform(
        RollPitchYaw(np.pi/2, 0, np.pi/2), np.array([0, 0, 0.114]))
    plant.WeldFrames(A=plant.GetFrameByName("iiwa_link_7"),
                     B=plant.GetFrameByName("body", schunk_model),
                     X_AB=X_L7E)

    plant.mutable_gravity_field().set_gravity_vector(gravity)

    # Add objects
    object_models_list = []
    for i, sdf_path in enumerate(object_sdf_paths):
        object_models_list.append(
            parser.AddModelFromFile(sdf_path, model_name="box%d" % i))

    plant.Finalize()

    return (plant,
            scene_graph,
            [robot_model, schunk_model],
            object_models_list)


def create_iiwa_plant_with_schunk_and_bin(
        builder, object_sdf_paths: List[str], time_step=5e-4):
    """
    :param builder: a DiagramBuilder object.
    :param object_sdf_paths: list of absolute paths to object.sdf files.
    :return:
    """
    # MultibodyPlant
    plant = MultibodyPlant(time_step)
    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)

    # Add bin
    X_WB = RigidTransform.Identity()
    X_WB.set_translation([0.6, 0, 0.])
    bin_sdf_path = FindResourceOrThrow(
        "drake/examples/manipulation_station/models/bin.sdf")
    parser.AddModelFromFile(bin_sdf_path)
    plant.WeldFrames(
        A=plant.world_frame(),
        B=plant.GetFrameByName("bin_base"),
        X_AB=X_WB)

    # fix robot to world
    robot_model = parser.AddModelFromFile(iiwa_sdf_path_drake)
    plant.WeldFrames(A=plant.world_frame(),
                     B=plant.GetFrameByName("iiwa_link_0"),
                     X_AB=RigidTransform.Identity())

    # fix schunk to l7
    schunk_sdf_path = FindResourceOrThrow(
          "drake/manipulation/models/wsg_50_description/sdf"
          "/schunk_wsg_50_ball_contact.sdf")
    schunk_model = parser.AddModelFromFile(schunk_sdf_path)
    X_L7E = RigidTransform(
        RollPitchYaw(np.pi/2, 0, np.pi/2), np.array([0, 0, 0.114]))
    plant.WeldFrames(A=plant.GetFrameByName("iiwa_link_7"),
                     B=plant.GetFrameByName("body", schunk_model),
                     X_AB=X_L7E)

    # Add objects
    object_models_list = []
    for i, sdf_path in enumerate(object_sdf_paths):
        object_models_list.append(
            parser.AddModelFromFile(sdf_path, model_name="box%d" % i))

    # Gravity.
    plant.mutable_gravity_field().set_gravity_vector(gravity)
    plant.Finalize()

    return (plant,
            scene_graph,
            [[robot_model, schunk_model]],
            object_models_list)


def create_2d_gripper_plant(builder, *args):
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
    gripper_sdf_path = os.path.join(model_dir_path, "models", "gripper.sdf")
    robot_model = parser.AddModelFromFile(gripper_sdf_path)

    # Add object
    object_sdf_path = os.path.join(model_dir_path, "models", "sphere_yz.sdf")
    object_model = parser.AddModelFromFile(object_sdf_path)

    plant.Finalize()

    return (plant,
            scene_graph,
            [robot_model],
            [object_model])
