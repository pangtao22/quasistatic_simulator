#%%
import copy

import numpy as np
from matplotlib import pyplot as plt

from pydrake.all import (ConnectMeshcatVisualizer,
    Simulator, SpatialForce, AbstractValue, BodyIndex)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.multibody.plant import (MultibodyPlant,
                                     ExternallyAppliedSpatialForce)
from pydrake.multibody.parsing import Parser
from pydrake.math import RigidTransform
from pydrake.systems.primitives import TrajectorySource, LogOutput

from setup_environments import (create_iiwa_plant, gravity)

from iiwa_controller.iiwa_controller.utils import (
    create_iiwa_controller_plant)
from iiwa_controller.iiwa_controller.robot_internal_controller import (
    RobotInternalController)

from contact_aware_control.plan_runner.plan_utils import (
    RenderSystemWithGraphviz)

from quasistatic_simulator import QuasistaticSimulator


#%%
class LoadApplier(LeafSystem):
    def __init__(self, F_WB_traj: PiecewisePolynomial, body_idx: BodyIndex):
        LeafSystem.__init__(self)
        self.set_name("load_applier")

        self.spatial_force_output_port = \
            self.DeclareAbstractOutputPort(
                "external_spatial_force",
                lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
                self.CalcOutput)

        self.F_WB_traj = F_WB_traj
        self.body_idx = body_idx

    def CalcOutput(self, context, spatial_forces_vector):
        t = context.get_time()

        easf = ExternallyAppliedSpatialForce()
        F = self.F_WB_traj.value(t).squeeze()
        easf.F_Bq_W = SpatialForce([0, 0, 0], F)
        easf.body_index = self.body_idx

        spatial_forces_vector.set_value([easf])


#%%
def run_sim(q_traj_iiwa: PiecewisePolynomial,
            Kp_iiwa: np.array,
            gravity: np.array,
            time_step):
    # Build diagram.
    builder = DiagramBuilder()

    # MultibodyPlant
    plant, scene_graph, robot_models, _ = create_iiwa_plant(
        builder, [], time_step)

    iiwa_model = robot_models[0]

    # IIWA controller
    plant_robot, _ = create_iiwa_controller_plant(gravity)
    controller_iiwa = RobotInternalController(
        plant_robot=plant_robot, joint_stiffness=Kp_iiwa,
        controller_mode="impedance")
    builder.AddSystem(controller_iiwa)
    builder.Connect(controller_iiwa.GetOutputPort("joint_torques"),
                    plant.get_actuation_input_port(iiwa_model))
    builder.Connect(plant.get_state_output_port(iiwa_model),
                    controller_iiwa.robot_state_input_port)

    # IIWA Trajectory source
    traj_source_iiwa = TrajectorySource(q_traj_iiwa)
    builder.AddSystem(traj_source_iiwa)
    builder.Connect(
        traj_source_iiwa.get_output_port(0),
        controller_iiwa.joint_angle_commanded_input_port)

    # meshcat visualizer
    viz = ConnectMeshcatVisualizer(
        builder, scene_graph, frames_to_draw={"iiwa": {"link_ee"}})

    # force on link 7.
    F_WB = np.zeros((2, 3))
    F_WB[1] = [0, 0, -100.]
    F_WB_traj = PiecewisePolynomial.FirstOrderHold(
        [0, q_traj_iiwa.end_time() / 2], F_WB.T)
    load_applier = LoadApplier(
        F_WB_traj,
        plant.GetBodyByName("iiwa_link_7").index())
    builder.AddSystem(load_applier)
    builder.Connect(
        load_applier.spatial_force_output_port,
        plant.get_applied_spatial_force_input_port())

    # Logs
    iiwa_log = LogOutput(plant.get_state_output_port(iiwa_model), builder)
    iiwa_log.set_publish_period(0.001)
    diagram = builder.Build()

    RenderSystemWithGraphviz(diagram)

    # %% Run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()
    context_controller = diagram.GetSubsystemContext(controller_iiwa, context)
    context_plant = diagram.GetSubsystemContext(plant, context)

    controller_iiwa.tau_feedforward_input_port.FixValue(
        context_controller, np.zeros(7))

    # robot initial configuration.
    q_iiwa_0 = q_traj_iiwa.value(0).squeeze()
    t_final = q_traj_iiwa.end_time()
    plant.SetPositions(context_plant, iiwa_model, q_iiwa_0)


    sim.Initialize()
    sim.set_target_realtime_rate(0)
    sim.AdvanceTo(t_final)

    return iiwa_log, controller_iiwa  # sim, plant, object_models


#%%
# run simulation
Kq_a = np.array([800., 600, 600, 600, 400, 200, 200])
q_a_initial_guess = np.array([0, 0, 0, -1.70, 0, 1.0, 0])

qa_knots = np.zeros((2, 7))
qa_knots[0] = q_a_initial_guess
qa_knots[1] = q_a_initial_guess

q_iiwa_traj = PiecewisePolynomial.FirstOrderHold([0, 2], qa_knots.T)

iiwa_log, controller_iiwa = run_sim(q_iiwa_traj, Kq_a, gravity, time_step=1e-5)

#%%
q_iiwa = iiwa_log.data().T[:, :7]
v_iiwa = iiwa_log.data().T[:, 7:]
t = iiwa_log.sample_times()
Kv = np.array(controller_iiwa.Kv_log)
tau_Kq = np.array(controller_iiwa.tau_stiffness_log)
tau_Kv = np.array(controller_iiwa.tau_damping_log)

#%% Quasistatic.
q_sim = QuasistaticSimulator(
    create_iiwa_plant,
    nd_per_contact=4,
    object_sdf_paths=[],
    joint_stiffness=Kq_a)

#%%
h = 0.2
q0_list = [q_iiwa_traj.value(0).squeeze()]
q_list = copy.deepcopy(q0_list)

q_a_log = []
q_a_cmd_log = []
q_log = []

n_steps = int(q_iiwa_traj.end_time() / h)

for i in range(n_steps):
    q_a_cmd = q_iiwa_traj.value(h * i).squeeze()
    q_a_cmd_list = [q_a_cmd]
    tau_u_ext_list = [None]
    dq_u_list, dq_a_list = q_sim.step_anitescu(
            q_list, q_a_cmd_list, tau_u_ext_list, h,
            is_planar=False,
            contact_detection_tolerance=0.005)

    # Update q
    q_sim.step_configuration(q_list, dq_u_list, dq_a_list, is_planar=False)
    q_sim.update_configuration(q_list)
    q_sim.draw_current_configuration()

    q_a_log.append(q_list[-1])
    q_a_cmd_log.append(q_a_cmd)
    q_log.append(copy.deepcopy(q_list))

    # time.sleep(h)
    # print("t = ", i * h)
    # input("step?")



#%%
nq = 7
fig, axes = plt.subplots(7, 1, figsize=(4, 10), dpi=150)
for i in range(nq):
    axes[i].plot(t, q_iiwa[:, i])
plt.show()
