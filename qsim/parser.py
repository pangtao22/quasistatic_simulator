import os
import copy
from collections import OrderedDict

import numpy as np
import parse
import yaml

from qsim_cpp import (QuasistaticSimulatorCpp,
                      BatchQuasistaticSimulator, GradientMode)

from .model_paths import package_paths_dict
from .simulator import QuasistaticSimulator, QuasistaticSimParameters
from .system import (QuasistaticSystem, QuasistaticSystemBackend)


class QuasistaticParser:
    def __init__(self, quasistatic_model_path: str):
        with open(quasistatic_model_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model_directive_path = self.parse_path(config['model_directive'])

        # robots
        robot_stiffness_dict = {}
        for robot in config['robots']:
            robot_stiffness_dict[robot['name']] = np.array(robot['Kp'],
                                                           dtype=float)
        self.robot_stiffness_dict = robot_stiffness_dict

        # objects
        object_sdf_paths = {}
        object_sdf_paths_ordered = OrderedDict()
        if config['objects'] is not None:
            for obj in config['objects']:
                name = obj['name']
                path = self.parse_path(obj['file'])
                object_sdf_paths[name] = path
                object_sdf_paths_ordered[name] = path
        self.object_sdf_paths = object_sdf_paths
        self.object_sdf_paths_ordered = object_sdf_paths_ordered

        # quasistatic_sim_params
        params = QuasistaticSimParameters()
        self.q_sim_params_dict = {
            name: params.__getattribute__(name)
            for name in params.__dir__() if not name.startswith("_")}

        self.set_sim_params(**copy.deepcopy(config['quasistatic_sim_params']))

    def set_sim_params(self, **kwargs):
        for name, value in kwargs.items():
            assert name in self.q_sim_params_dict.keys()
            self.q_sim_params_dict[name] = value

    def make_params_struct(self):
        sim_params = QuasistaticSimParameters()
        for name, value in self.q_sim_params_dict.items():
            sim_params.__setattr__(name, value)

        return sim_params

    def parse_path(self, model_path: str):
        """
        A model_path read from the yaml file should have the format
            package://package_name/file_name
        """
        package_name, file_name = parse.parse("package://{}/{}", model_path)
        return os.path.join(package_paths_dict[package_name], file_name)

    def set_quasi_dynamic(self, is_quasi_dynamic: bool):
        """
        Set self.q_sim_params.is_quasi_dynamic to the input is_quasi_dynamic,
            the default value is False.
        """
        self.q_sim_params_dict['is_quasi_dynamic'] = is_quasi_dynamic

    def get_gravity(self):
        return np.array(self.q_sim_params_dict['gravity'])

    def get_param(self, name: str):
        return copy.deepcopy(self.q_sim_params_dict[name])

    def get_robot_stiffness_by_name(self, name: str):
        return np.array(self.robot_stiffness_dict[name])

    def make_system(self, backend: QuasistaticSystemBackend):
        q_sim_params = self.make_params_struct()
        QuasistaticSimulator.check_params_validity(q_sim_params)
        return QuasistaticSystem(model_directive_path=self.model_directive_path,
                                 robot_stiffness_dict=self.robot_stiffness_dict,
                                 object_sdf_paths=self.object_sdf_paths,
                                 sim_params=q_sim_params, backend=backend)

    def make_simulator_py(self, internal_vis: bool):
        q_sim_params = self.make_params_struct()
        QuasistaticSimulator.check_params_validity(q_sim_params)
        return QuasistaticSimulator(
            model_directive_path=self.model_directive_path,
            robot_stiffness_dict=self.robot_stiffness_dict,
            object_sdf_paths=self.object_sdf_paths_ordered,
            sim_params=q_sim_params,
            internal_vis=internal_vis)

    def make_simulator_cpp(self):
        q_sim_params = self.make_params_struct()
        QuasistaticSimulator.check_params_validity(q_sim_params)
        return QuasistaticSimulatorCpp(
            model_directive_path=self.model_directive_path,
            robot_stiffness_str=self.robot_stiffness_dict,
            object_sdf_paths=self.object_sdf_paths,
            sim_params=q_sim_params)

    def make_batch_simulator(self):
        q_sim_params = self.make_params_struct()
        QuasistaticSimulator.check_params_validity(q_sim_params)
        return BatchQuasistaticSimulator(
            model_directive_path=self.model_directive_path,
            robot_stiffness_str=self.robot_stiffness_dict,
            object_sdf_paths=self.object_sdf_paths,
            sim_params=q_sim_params)
