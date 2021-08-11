from typing import Dict, List

import numpy as np
from quasistatic_simulator.examples.model_paths import add_package_paths_local
from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder,
                         AddMultibodyPlantSceneGraph,
                         ProcessModelDirectives, LoadModelDirectives)


def get_rotation_matrix_from_normal(normal):
    R = np.eye(3)
    R[:, 2] = normal
    if np.linalg.norm(normal[:2]) < 1e-6:
        R[:, 0] = [0, normal[2], -normal[1]]
    else:
        R[:, 0] = [normal[1], -normal[0], 0]
    R[:, 0] /= np.linalg.norm(R[:, 0])
    R[:, 1] = np.cross(normal, R[:, 0])

    return R


def calc_tangent_vectors(normal, nd):
    normal = normal.copy()
    normal /= np.linalg.norm(normal)
    if nd == 2:
        # Makes sure that dC is in the yz plane.
        dC = np.zeros((2, 3))
        dC[0] = np.cross(np.array([1, 0, 0]), normal)
        dC[1] = -dC[0]
    else:
        R = get_rotation_matrix_from_normal(normal)
        dC = np.zeros((nd, 3))

        for i in range(nd):
            theta = 2 * np.pi / nd * i
            dC[i] = [np.cos(theta), np.sin(theta), 0]

        dC = (R.dot(dC.T)).T
    return dC


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

    # Objects
    object_models = set()
    for name, sdf_path in object_sdf_paths.items():
        object_models.add(
            parser.AddModelFromFile(sdf_path, model_name=name))

    # Robots
    robot_models = set()
    for name in robot_names:
        robot_model = plant.GetModelInstanceByName(name)
        robot_models.add(robot_model)

    # gravity
    plant.mutable_gravity_field().set_gravity_vector(gravity)
    plant.Finalize()

    return plant, scene_graph, robot_models, object_models
