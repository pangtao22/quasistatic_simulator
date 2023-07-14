from enum import IntEnum
from typing import Dict, Set, List

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
    ContactResults,
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
    ):
        """
        Contact force visualization is only supported by the C++ visualizer.
        """
        self.q_sys = q_sys
        self.visualization_type = visualization_type
        self.q_sim = q_sys.q_sim

        self.draw_forces = visualization_type == QsimVisualizationType.Cpp

        builder = DiagramBuilder()
        builder.AddSystem(q_sys)

        if visualization_type == QsimVisualizationType.Cpp:
            self.meshcat = StartMeshcat()
            self.meshcat_vis = MeshcatVisualizer.AddToBuilder(
                builder, q_sys.query_object_output_port, self.meshcat
            )

            # ContactVisualizer
            self.contact_vis = ContactVisualizer.AddToBuilder(
                builder, q_sys.contact_results_output_port, self.meshcat
            )
        elif visualization_type == QsimVisualizationType.Python:
            self.meshcat_vis = ConnectMeshcatVisualizerPy(
                builder, output_port=q_sys.query_object_output_port
            )
            self.contact_vis = None

        self.diagram = builder.Build()

        (
            self.context,
            self.context_q_sys,
            self.context_meshcat,
            self.context_contact_vis,
        ) = self.create_context()

        if visualization_type == QsimVisualizationType.Python:
            self.meshcat_vis.load(self.context_meshcat)

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

    def create_context(self):
        context = self.diagram.CreateDefaultContext()
        context_q_sys = self.q_sys.GetMyMutableContextFromRoot(context)
        context_meshcat = self.meshcat_vis.GetMyMutableContextFromRoot(context)

        context_contact_vis = None
        if self.contact_vis:
            context_contact_vis = self.contact_vis.GetMyMutableContextFromRoot(
                context
            )

        return (
            context,
            context_q_sys,
            context_meshcat,
            context_contact_vis,
        )

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

    def draw_configuration(
        self, q: np.ndarray, contact_results: ContactResults | None = None
    ):
        """
        If self.visualization_type == QsimVisualizationType.Python,
         contact_results is silently ignored.
        """
        self.context_q_sys.SetDiscreteState(q)
        if self.visualization_type == QsimVisualizationType.Python:
            self.meshcat_vis.DoPublish(self.context_meshcat, [])
        else:
            # CPP meshcat.
            # Body poses
            self.meshcat_vis.ForcedPublish(self.context_meshcat)

            # Contact forces
            if contact_results:
                self.contact_vis.GetInputPort("contact_results").FixValue(
                    self.context_contact_vis,
                    AbstractValue.Make(contact_results),
                )
                self.contact_vis.ForcedPublish(self.context_contact_vis)

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

    def publish_trajectory(
        self,
        h: float,
        q_knots: np.ndarray,
        contact_results_list: List[ContactResults] | None = None,
    ):
        """
        q_knots: (T + 1, n_q) array.
        contact_results_list: (T,) list.
        For 1 <= i <= T, contact_results_list[i - 1] is the contact forces at
         configuration q_knots[i].

        contact_results_list is silently ignored if visualization_type is
         python.
        """
        if contact_results_list:
            n_contacts = len(contact_results_list)
            n_knots = len(q_knots)
            assert n_knots == n_contacts + 1 or n_contacts == 0

        if self.visualization_type == QsimVisualizationType.Cpp:
            self.contact_vis.Delete()
            self.meshcat_vis.DeleteRecording()
            self.meshcat_vis.StartRecording(False)
            for i, t in enumerate(np.arange(len(q_knots)) * h):
                self.context.SetTime(t)
                contact_results = None
                if contact_results_list and i > 0:
                    contact_results = contact_results_list[i - 1]

                self.draw_configuration(
                    q=q_knots[i], contact_results=contact_results
                )

            self.meshcat_vis.StopRecording()
            self.meshcat_vis.PublishRecording()
        elif self.visualization_type == QsimVisualizationType.Python:
            self.meshcat_vis.draw_period = h
            self.meshcat_vis.reset_recording()
            self.meshcat_vis.start_recording()
            for q_i in q_knots:
                self.draw_configuration(q_i)

            self.meshcat_vis.stop_recording()
            self.meshcat_vis.publish_recording()
