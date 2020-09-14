import os
import pathlib

import numpy as np

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform

#%%
box_sdf_path = os.path.join("models", "box.sdf")

plant = MultibodyPlant(1e-3)
parser = Parser(plant=plant)
parser.AddModelFromFile(box_sdf_path)

plant.Finalize()

#%%
context = plant.CreateDefaultContext()
body_frame = plant.GetFrameByName("base_link")
angle = 0.5
axis = np.array([1, 2, 3.])
axis /= np.linalg.norm(axis)
Q_WB = Quaternion(np.hstack([np.cos(angle / 2), np.sin(angle / 2) * axis]))
plant.SetPositions(context, np.hstack([Q_WB.wxyz(), np.zeros(3)]))

J = plant.CalcJacobianSpatialVelocity(
    context=context,
    with_respect_to=JacobianWrtVariable.kV,
    frame_B=body_frame,
    p_BP=np.zeros(3),
    frame_A=plant.world_frame(),
    frame_E=plant.world_frame())

print(J)

