import os

import pydrake
from robotics_utilities.iiwa_controller.utils import get_package_path
from pydrake.all import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser

models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
box3d_medium_sdf_path = os.path.join(models_dir, "box_0.6m.sdf")
box3d_small_sdf_path = os.path.join(models_dir, "box_0.5m.sdf")
box3d_8cm_sdf_path = os.path.join(models_dir, "box_0.08m.sdf")
box3d_7cm_sdf_path = os.path.join(models_dir, "box_0.07m.sdf")
box3d_6cm_sdf_path = os.path.join(models_dir, "box_0.06m.sdf")
sphere2d_big_sdf_path = os.path.join(models_dir, "sphere_yz_rotation_big.sdf")
# Model package paths.
drake_manipulation_models_path = os.path.join(
    pydrake.common.GetDrakePath(), "manipulation/models"
)
iiwa_controller_models_path = os.path.join(get_package_path(), "models")
package_paths_dict = {
    "quasistatic_simulator": models_dir,
    "drake_manipulation_models": drake_manipulation_models_path,
    "iiwa_controller": iiwa_controller_models_path,
}


def add_package_paths_local(parser: Parser):
    for name, path in package_paths_dict.items():
        parser.package_map().Add(name, path)
