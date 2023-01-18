import networkx
import numpy as np
import plotly.graph_objects as go

from pydrake.all import RigidTransform, RollPitchYaw
import meshcat


# %% meshcat
def set_orthographic_camera_yz(vis: meshcat.Visualizer) -> None:
    # use orthographic camera, show YZ plane.
    camera = meshcat.geometry.OrthographicCamera(
        left=-1.2, right=1.2, bottom=-0.8, top=1.0, near=-1000, far=1000
    )
    vis["/Cameras/default/rotated"].set_object(camera)
    vis["/Cameras/default/rotated/<object>"].set_property("position", [0, 0, 0])
    vis["/Cameras/default"].set_transform(
        meshcat.transformations.translation_matrix([1, 0, 0])
    )
