import os

import pydrake
from iiwa_controller.iiwa_controller.utils import get_package_path
from pydrake.all import (MultibodyPlant, AddMultibodyPlantSceneGraph)
from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser

models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
box2d_big_sdf_path = os.path.join(models_dir, "box_yz_rotation_big.sdf")
box3d_big_sdf_path = os.path.join(models_dir, "box_1m.sdf")
box3d_medium_sdf_path = os.path.join(models_dir, "box_0.6m.sdf")
box3d_small_sdf_path = os.path.join(models_dir, "box_0.5m.sdf")
box3d_8cm_sdf_path = os.path.join(models_dir, "box_0.08m.sdf")
box3d_7cm_sdf_path = os.path.join(models_dir, "box_0.07m.sdf")
box3d_6cm_sdf_path = os.path.join(models_dir, "box_0.06m.sdf")
sphere2d_big_sdf_path = os.path.join(models_dir,
                                     "sphere_yz_rotation_big.sdf")


def add_package_paths_local(parser: Parser):
    parser.package_map().Add(
        "drake_manipulation_models",
        os.path.join(pydrake.common.GetDrakePath(),
                     "manipulation/models"))

    parser.package_map().Add("quasistatic_simulator", models_dir)

    parser.package_map().Add(
        'iiwa_controller', os.path.join(get_package_path(), 'models'))


#TODO: delete this after adding unit test for 2d gripper example.
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
    gripper_sdf_path = os.path.join(models_dir, "models", "gripper.sdf")
    robot_model = parser.AddModelFromFile(gripper_sdf_path)

    # Add object
    object_sdf_path = os.path.join(models_dir, "models", "sphere_yz.sdf")
    object_model = parser.AddModelFromFile(object_sdf_path)

    plant.Finalize()

    return (plant,
            scene_graph,
            [robot_model],
            [object_model])
