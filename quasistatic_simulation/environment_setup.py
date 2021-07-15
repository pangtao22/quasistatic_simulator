from typing import Dict, List

import numpy as np
from examples.model_paths import add_package_paths_local
from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder,
                         AddMultibodyPlantSceneGraph,
                         ProcessModelDirectives, LoadModelDirectives)


def create_plant_with_robots_and_objects(builder: DiagramBuilder,
                                         model_directive_path: str,
                                         robot_names: List[str],
                                         object_sdf_paths: Dict[str, str],
                                         time_step: float, gravity: np.ndarray):
    """
    Add plant and scene_graph constructed from a model_directive to builder.
    :param builder:
    :param model_directive_path:
    :param robot_names: names in this list must be consistent with the
        corresponding model directive .yml file.
    :param object_names:
    :param time_step:
    :param gravity:
    :return:
    """

    # MultibodyPlant
    plant = MultibodyPlant(time_step)
    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    add_package_paths_local(parser)
    ProcessModelDirectives(LoadModelDirectives(model_directive_path),
                           plant, parser)

    # Robots
    robot_models_list = []
    for name in robot_names:
        robot_model = plant.GetModelInstanceByName(name)
        robot_models_list.append(robot_model)

    # Objects
    object_models_list = []
    for name, sdf_path in object_sdf_paths.items():
        object_models_list.append(
            parser.AddModelFromFile(sdf_path, model_name=name))

    # gravity
    plant.mutable_gravity_field().set_gravity_vector(gravity)
    plant.Finalize()

    return plant, scene_graph, robot_models_list, object_models_list
