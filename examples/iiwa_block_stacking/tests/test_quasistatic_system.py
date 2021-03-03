import unittest

from examples.setup_simulation_diagram import run_quasistatic_sim
from examples.iiwa_block_stacking.run_manual_quasistatic import *


class TestQuasistaticSystem(unittest.TestCase):
    def test_quasistatic_system(self):
        """
        The order in which drake's system framework's discrete update events,
            publishing events and logging happens are scheduled can be tricky
            to get right. This test makes sure that QuasistaticSystem,
            the LeafSystem wrapper of QuasistaticSimulator,
            is implemented correctly, by comparing the simulation results from
            using Simulator, and manually calling
            QuasistaticSimualtor.step_anitescu in a for loop.
            The two simulation results should be identical, but in practice
            there is a small, non-deterministic(!) difference. The
            non-deterministic-ness could be due to the QP not having a
            unique solution.

        Here's what I think happens, in drake's discrete system updates,
            based on drake's online documentation:
            https://drake.mit.edu/doxygen_cxx/group__discrete__systems.html

        A discrete system can be written as
            x(l + 1) = f(x(l), u(l)).
        For quasistatic system, x(l) is the system configuration at step l,
            and u(l) is the commanded robot positions at step (l + 1). This
            is why commanded trajectories (q_a_traj's) are shifted to start at
            -h, where h is the simulation time step. With the shifted
            q_a_traj, inputs at step l can be computed as
            u(l) = q_a_shifted(l * h) = q_a((l+1) * h).

        Also note that drake's update for step l+1 happens at step l,
            which means that in the interval [l*h, (l+1)*h), the state x has
            value x(l+1).

        Let x(l-) and x(l+) denote the state before and after an update
            event, respectively. According to the update rule above,
            x(l-) = x(l), x(l+) = x(l+1).

        If a publish event and a discrete update event are scheduled at the
            same time step l, then the publish event sees x(l-). Similarly,
            if logging of an output port that depends on the state is also
            scheduled at l, then x(l-) is logged.

        TL;DR: drake's SignalLogger at t = (l * h) logs x(l).
        """

        # Simulation time step.
        h = 0.2

        # Simulate using Simulator.
        loggers_dict_systems_str, q_sys = run_quasistatic_sim(
            q_a_traj_dict_str=q_a_traj_dict_str,
            q0_dict_str=q0_dict_str,
            robot_info_dict=robot_info_dict,
            object_sdf_paths=object_sdf_paths_dict,
            h=h,
            gravity=gravity,
            is_visualizing=False,
            real_time_rate=0.0)

        # Simulate manually.
        q_logs_dict_str, t_quasistatic = \
            run_quasistatic_sim_manually(h=0.2, is_visualizing=False,
                                         gravity=gravity)

        tolerance = 5e-5
        for model_name in q0_dict_str.keys():
            q_log_system = loggers_dict_systems_str[model_name].data().T
            q_log_manual = q_logs_dict_str[model_name]
            # print(model_name, np.linalg.norm(q_log_system - q_log_manual))
            self.assertLessEqual(
                np.linalg.norm(q_log_system - q_log_manual), tolerance,
                "Trajectory difference between manual and Systems simulations "
                "is larger than {} for model instance named {}.".format(
                    tolerance, model_name))


if __name__ == '__main__':
    unittest.main()
