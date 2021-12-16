import os

import numpy as np
import yaml

from .simulator import QuasistaticSimulator, QuasistaticSimParameters
from .system import QuasistaticSystem, QuasistaticSystemBackend
from .model_paths import models_dir


class QuasistaticParser:
    def __init__(self, quasistatic_model_path: str):
        with open(quasistatic_model_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_directive_path = os.path.join(
            models_dir, config['model_directive_name'])
        # robots
        robot_stiffness_dict = {}
        for robot in config['robots']:
            robot_stiffness_dict[robot['name']] = np.array(robot['Kp'],
                                                           dtype=float)
        self.robot_stiffness_dict = robot_stiffness_dict

        # objects
        object_sdf_paths = {}
        for obj in config['objects']:
            object_sdf_paths[obj['name']] = os.path.join(
                models_dir, obj['file_name'])
        self.object_sdf_paths = object_sdf_paths

        # quasistatic_sim_params
        q_sim_params = config['quasistatic_sim_params']
        d = q_sim_params['contact_detection_tolerance']
        if type(d) is str:
            assert d == 'inf'
            d = np.inf

        self.q_sim_params = QuasistaticSimParameters(
            gravity=np.array(q_sim_params['gravity'], dtype=float),
            nd_per_contact=q_sim_params['nd_per_contact'],
            contact_detection_tolerance=d,
            is_quasi_dynamic=q_sim_params['is_quasi_dynamic'])

    def get_gravity(self):
        return np.array(self.q_sim_params.gravity)

    def get_robot_stiffness_by_name(self, name: str):
        return np.array(self.robot_stiffness_dict[name])

    def make_simulator(self):
        pass

    def make_system(self, time_step: float, backend: QuasistaticSystemBackend):
        return QuasistaticSystem(
            time_step=time_step,
            model_directive_path=self.model_directive_path,
            robot_stiffness_dict=self.robot_stiffness_dict,
            object_sdf_paths=self.object_sdf_paths,
            sim_params=self.q_sim_params,
            backend=backend)
