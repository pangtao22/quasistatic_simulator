import os

import numpy as np
import parse
import yaml

from qsim_cpp import (QuasistaticSimParametersCpp, QuasistaticSimulatorCpp,
                      GradientMode)

from .model_paths import package_paths_dict
from .simulator import QuasistaticSimulator, QuasistaticSimParameters
from .system import (QuasistaticSystem, QuasistaticSystemBackend)
from .utils import cpp_params_from_py_params


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
        for obj in config['objects']:
            object_sdf_paths[obj['name']] = self.parse_path(obj['file'])
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
            contact_detection_tolerance=d)

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
        param_old = self.q_sim_params
        self.q_sim_params = QuasistaticSimParameters(
            gravity=param_old.gravity,
            nd_per_contact=param_old.nd_per_contact,
            contact_detection_tolerance=param_old.contact_detection_tolerance,
            is_quasi_dynamic=is_quasi_dynamic,
            mode=param_old.mode,
            log_barrier_weight=param_old.log_barrier_weight,
            gradient_mode=param_old.gradient_mode,
            grad_from_active_constraints=param_old.grad_from_active_constraints)

    def get_gravity(self):
        return np.array(self.q_sim_params.gravity)

    def get_robot_stiffness_by_name(self, name: str):
        return np.array(self.robot_stiffness_dict[name])

    def make_system(self, time_step: float, backend: QuasistaticSystemBackend):
        self.check_params_validity(self.q_sim_params)
        return QuasistaticSystem(
            time_step=time_step,
            model_directive_path=self.model_directive_path,
            robot_stiffness_dict=self.robot_stiffness_dict,
            object_sdf_paths=self.object_sdf_paths,
            sim_params=self.q_sim_params,
            backend=backend)

    def make_simulator_py(self, internal_vis: bool):
        self.check_params_validity(self.q_sim_params)
        return QuasistaticSimulator(
            model_directive_path=self.model_directive_path,
            robot_stiffness_dict=self.robot_stiffness_dict,
            object_sdf_paths=self.object_sdf_paths,
            sim_params=self.q_sim_params,
            internal_vis=internal_vis)

    def make_simulator_cpp(self):
        self.check_params_validity(self.q_sim_params)
        return QuasistaticSimulatorCpp(
            model_directive_path=self.model_directive_path,
            robot_stiffness_str=self.robot_stiffness_dict,
            object_sdf_paths=self.object_sdf_paths,
            sim_params=cpp_params_from_py_params(self.q_sim_params))

    @staticmethod
    def check_params_validity(q_params: QuasistaticSimParameters):
        if (q_params.nd_per_contact > 2 and q_params.gradient_mode ==
                GradientMode.kAB):
            raise RuntimeError("Computing A matrix for 3D systems is not yet "
                               "supported.")
