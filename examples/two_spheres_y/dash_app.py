import copy
import json

import dash
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from matplotlib import cm
from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix

from dash_common import (
    set_orthographic_camera_yz,
)

from qsim.parser import QuasistaticParser, QuasistaticSystemBackend
from qsim.simulator import (
    ForwardDynamicsMode,
    GradientMode,
    InternalVisualizationType,
)
from sim_setup import *

#%% make q_sim
q_parser = QuasistaticParser(os.path.join(models_dir, q_model_path))
q_sim = q_parser.make_simulator_cpp()
q_sim_batch = q_parser.make_batch_simulator()
q_sim_py = q_parser.make_simulator_py(
    internal_vis=InternalVisualizationType.Python
)
meshcat_vis = q_sim_py.viz.vis
set_orthographic_camera_yz(meshcat_vis)

plant = q_sim.get_plant()
idx_a = plant.GetModelInstanceByName(robot_name)
idx_u = plant.GetModelInstanceByName(object_name)
idx_a_into_q = q_sim.get_q_a_indices_into_q()
idx_u_into_q = q_sim.get_q_u_indices_into_q()


#%% points at which dynamics is evaluated.
q0_dict = {idx_a: np.array([-1.0]), idx_u: np.array([0.2])}
q0 = q_sim.get_q_vec_from_dict(q0_dict)

y_u_min = -1.0
y_u_max = 1.0
r = 0.1
n_samples = 101

q_batch = np.tile(q0, (n_samples, 1))
u_batch = np.linspace(y_u_min, y_u_max, n_samples)[:, None]

#%% prepare data for visualization dynamics evaluations
sim_params = copy.deepcopy(q_sim.get_sim_params())
sim_params.unactuated_mass_scale = 10.0
sim_params.h = 0.1
q_next_batch_dict = {}

for kappa_log in [0, 1, 2, 3, 4]:
    if kappa_log != 5:
        sim_params.forward_mode = ForwardDynamicsMode.kLogPyramidMp
        sim_params.log_barrier_weight = 10**kappa_log
    else:
        sim_params.forward_mode = ForwardDynamicsMode.kQpMp

    q_next_batch_dict[kappa_log] = q_sim_batch.calc_dynamics_parallel(
        q_batch, u_batch, sim_params
    )[0]


#%% dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="dynamics-plot"),
                    width={"size": 6, "offset": 0, "order": 0},
                ),
                dbc.Col(
                    html.Iframe(
                        src=meshcat_vis.url(),
                        height=600,
                        width=800,
                    ),
                    width={"size": 6, "offset": 0, "order": 0},
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("u (commanded ball position)"),
                        dcc.Slider(
                            id="u",
                            min=y_u_min,
                            max=y_u_max,
                            value=-1,
                            step=(y_u_max - y_u_min) / (n_samples - 1),
                            marks={
                                -1: {"label": "-1"},
                                1: {"label": "1"},
                            },
                            tooltip={
                                "placement": "bottom",
                                "always_visible": True,
                            },
                        ),
                        html.H3("kappa (smoothing)"),
                        dcc.Slider(
                            id="kappa",
                            min=0,
                            max=4,
                            value=4,
                            step=None,
                            marks={
                                0: "1",
                                1: "10",
                                2: "100",
                                3: "1000",
                                4: "inf",
                            },
                        ),
                    ],
                    width={"size": 6, "offset": 0, "order": 0},
                ),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("dynamics-plot", "figure"),
    [Input("u", "value"), Input("kappa", "value")],
)
def q_a_slider_callback(u_value, kappa_value_log):
    y_u_step = (y_u_max - y_u_min) / (n_samples - 1)
    idx = max(2, int((u_value - y_u_min) / y_u_step))

    q_next_batch = q_next_batch_dict[kappa_value_log]

    q_sim_py.update_mbp_positions_from_vector(q_next_batch[idx])
    q_sim_py.draw_current_configuration()

    fig = px.scatter(
        x=q_next_batch[:idx, idx_a_into_q].squeeze(),
        y=q_next_batch[:idx, idx_u_into_q].squeeze(),
    )
    fig.update_layout(
        xaxis_range=[-1.2, 1.2],
        yaxis_range=[-0.2, 1.2],
        xaxis_title="u (commanded ball position)",
        yaxis_title="q_u (box position)",
        scene=dict(aspectratio=dict(x=1.0, y=1.0)),
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
