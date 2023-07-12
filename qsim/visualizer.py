from enum import IntEnum
import os
from typing import Dict, Set, List

from matplotlib import cm
import numpy as np
from pydrake.all import (
    ModelInstanceIndex,
    MultibodyPlant,
    RigidTransform,
    DiagramBuilder,
    StartMeshcat,
    ContactVisualizer,
    AbstractValue,
    MeshcatVisualizer,
)

from manipulation.meshcat_utils import AddMeshcatTriad

from qsim.system import QuasistaticSystem
from qsim.meshcat_visualizer_old import AddTriad

from .meshcat_visualizer_old import (
    ConnectMeshcatVisualizer as ConnectMeshcatVisualizerPy,
)


class QsimVisualizationType(IntEnum):
    """
    We need to keep the python MeshcatVisualizer around, because plotly RRT
    visualizer does not work with drake's CPP-based MeshcatVisualizer.
    """

    Cpp = 1
    Python = 2


class QuasistaticVisualizer:
    def __init__(
        self,
        q_sys: QuasistaticSystem,
        visualization_type: QsimVisualizationType,
        draw_forces: bool,
    ):
        self.q_sys = q_sys
        self.visualization_type = visualization_type
        self.draw_forces = draw_forces

        self.q_sim = q_sys.q_sim

        plant = q_sys.plant
        scene_graph = self.q_sim.get_scene_graph()

        builder = DiagramBuilder()
        builder.AddSystem(q_sys)

        if visualization_type == QsimVisualizationType.Cpp:
            self.meshcat = StartMeshcat()
            self.meshcat_vis = MeshcatVisualizer.AddToBuilder(
                builder, q_sys.query_object_output_port, self.meshcat
            )
            if draw_forces:
                # ContactVisualizer
                self.contact_viz = ContactVisualizer.AddToBuilder(
                    builder, plant, self.meshcat
                )
        elif visualization_type == QsimVisualizationType.Python:
            self.meshcat_vis = ConnectMeshcatVisualizerPy(builder, scene_graph)
            self.meshcat_vis.load()
            self.contact_viz = None

        self.diagram = builder.Build()

        self.context = self.diagram.CreateDefaultContext()
        self.context_q_sys = self.q_sys.GetMyMutableContextFromRoot(
            self.context
        )
        self.context_meshcat = self.meshcat_vis.GetMyMutableContextFromRoot(
            self.context
        )

        if draw_forces:
            self.context_contact_vis = (
                self.contact_viz.GetMyMutableContextFromRoot(self.context)
            )

        self.plant = self.q_sim.get_plant()
        self.body_id_meshcat_name_map = self.get_body_id_to_meshcat_name_map()

    def get_body_id_to_meshcat_name_map(self):
        body_id_meshcat_name_map = {}
        prefix = "drake/plant"
        for model in self.q_sim.get_actuated_models():
            body_indices = self.plant.GetBodyIndices(model)
            model_name = self.plant.GetModelInstanceName(model)
            for bi in body_indices:
                body_name = self.plant.get_body(bi).name()
                name = prefix + f"/{model_name}/{body_name}"
                body_id_meshcat_name_map[bi] = name

        return body_id_meshcat_name_map

    @staticmethod
    def check_plants(
        plant_a: MultibodyPlant,
        plant_b: MultibodyPlant,
        models_all_a: Set[ModelInstanceIndex],
        models_all_b: Set[ModelInstanceIndex],
        velocity_indices_a: Dict[ModelInstanceIndex, np.ndarray],
        velocity_indices_b: Dict[ModelInstanceIndex, np.ndarray],
    ):
        """
        Make sure that plant_a and plant_b are identical.
        """
        assert models_all_a == models_all_b
        for model in models_all_a:
            name_a = plant_a.GetModelInstanceName(model)
            name_b = plant_b.GetModelInstanceName(model)
            assert name_a == name_b

            idx_a = velocity_indices_a[model]
            idx_b = velocity_indices_b[model]
            assert idx_a == idx_b

    def draw_configuration(self, q: np.ndarray):
        self.context_q_sys.SetDiscreteState(q)
        if self.visualization_type == QsimVisualizationType.Python:
            self.meshcat_vis.DoPublish(self.context_meshcat, [])
        else:
            # CPP meshcat.
            # Body poses
            self.meshcat_vis.ForcedPublish(self.context_meshcat)

            # Contact forces
            if self.draw_forces:
                self.contact_viz.GetInputPort("contact_results").FixValue(
                    self.context_contact_vis,
                    AbstractValue.Make(self.contact_results),
                )
                self.contact_viz.ForcedPublish(self.context_meshcat_contact)

    def draw_goal_triad(
        self,
        length: float,
        radius: float,
        opacity: float,
        X_WG: RigidTransform,
        name: str = "goal",
    ):
        if self.visualization_type == QsimVisualizationType.Cpp:
            AddMeshcatTriad(
                meshcat=self.meshcat,
                path=f"{name}/frame",
                length=length,
                radius=radius,
                opacity=opacity,
                X_PT=X_WG,
            )
        elif self.visualization_type == QsimVisualizationType.Python:
            AddTriad(
                vis=self.meshcat_vis.vis,
                name="frame",
                prefix=name,
                length=length,
                radius=radius,
                opacity=opacity,
            )
            self.meshcat_vis.vis[name].set_transform(X_WG.GetAsMatrix4())

        return name

    def draw_object_triad(
        self,
        length: float,
        radius: float,
        opacity: float,
        path: str,
    ):
        if self.visualization_type == QsimVisualizationType.Cpp:
            AddMeshcatTriad(
                meshcat=self.meshcat,
                path=f"visualizer/{path}/frame",
                length=length,
                radius=radius,
                opacity=opacity,
            )
        elif self.visualization_type == QsimVisualizationType.Python:
            AddTriad(
                vis=self.meshcat_vis.vis,
                name="frame",
                prefix=f"drake/plant/{path}",
                length=length,
                radius=radius,
                opacity=opacity,
            )

    def publish_trajectory(self, x_knots: np.ndarray, h: float):
        if self.visualization_type == QsimVisualizationType.Cpp:
            self.meshcat_vis.DeleteRecording()
            self.meshcat_vis.StartRecording(False)
            for i, t in enumerate(np.arange(len(x_knots)) * h):
                self.context.SetTime(t)
                self.context_q_sys.SetDiscreteState(x_knots[i])
                self.meshcat_vis.ForcedPublish(self.context_meshcat)

            self.meshcat_vis.StopRecording()
            self.meshcat_vis.PublishRecording()
        elif self.visualization_type == QsimVisualizationType.Python:
            self.meshcat_vis.draw_period = h
            self.meshcat_vis.reset_recording()
            self.meshcat_vis.start_recording()
            for x_i in x_knots:
                self.draw_configuration(x_i)

            self.meshcat_vis.stop_recording()
            self.meshcat_vis.publish_recording()
