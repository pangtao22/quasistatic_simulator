# -*- mode: cmake -*-
# vi: set ft=cmake :

# Copyright (c) 2017, Massachusetts Institute of Technology.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

add_library(get_model_paths get_model_paths.h get_model_paths.cc)
target_link_libraries(get_model_paths drake::drake)

add_library(optimization_derivatives qp_derivatives.h qp_derivatives.cc
        socp_derivatives.h socp_derivatives.cc)
target_link_libraries(optimization_derivatives drake::drake)

add_library(contact_computer contact_jacobian_calculator.h
        contact_jacobian_calculator.cc quasistatic_sim_params.h)
target_link_libraries(contact_computer drake::drake)

add_library(log_barrier_solver log_barrier_solver.h log_barrier_solver.cc)
target_link_libraries(log_barrier_solver drake::drake)

add_library(quasistatic_simulator
        quasistatic_simulator.h
        quasistatic_simulator.cc
        batch_quasistatic_simulator.h
        batch_quasistatic_simulator.cc
        quasistatic_parser.h
        quasistatic_parser.cc
        finite_differencing_gradient.h
        finite_differencing_gradient.cc)
target_link_libraries(quasistatic_simulator optimization_derivatives
        drake::drake get_model_paths contact_computer log_barrier_solver
        yaml-cpp)

pybind11_add_module(qsim_cpp MODULE qsim_cpp.cc)
target_link_libraries(qsim_cpp PUBLIC quasistatic_simulator)
set_target_properties(qsim_cpp PROPERTIES CXX_VISIBILITY_PRESET default)

add_executable(run_planar_hand_ball run_planar_hand_ball.cc)
target_link_libraries(run_planar_hand_ball quasistatic_simulator)

add_executable(run_3link_arm_3d run_3link_arm_3d.cc)
target_link_libraries(run_3link_arm_3d quasistatic_simulator)

add_executable(run_iiwa_box_stacking run_iiwa_box_stacking.cc)
target_link_libraries(run_iiwa_box_stacking quasistatic_simulator)

add_executable(run_allegro_hand_jacobian run_allegro_hand_jacobian.cc)
target_link_libraries(run_allegro_hand_jacobian quasistatic_simulator)

add_executable(run_allegro_hand_A run_allegro_hand_A.cc)
target_link_libraries(run_allegro_hand_A quasistatic_simulator)

add_executable(test_batch_simulator test_batch_simulator.cc)
target_link_libraries(test_batch_simulator quasistatic_simulator gtest)

add_executable(test_log_barrier_solver test_log_barrier_solver.cc)
target_link_libraries(test_log_barrier_solver log_barrier_solver gtest)

add_executable(test_contact_forces test_contact_forces.cc)
target_link_libraries(test_contact_forces quasistatic_simulator gtest)

add_executable(test_quasistatic_sim_gradients test_quasistatic_sim_gradients.cc)
target_link_libraries(test_quasistatic_sim_gradients
        quasistatic_simulator gtest)

add_test(NAME test_batch_simulator COMMAND test_batch_simulator)
add_test(NAME test_log_barrier_solver COMMAND test_log_barrier_solver)
add_test(NAME test_contact_forces COMMAND test_contact_forces)
add_test(NAME test_quasistatic_sim_gradients
         COMMAND test_quasistatic_sim_gradients)
