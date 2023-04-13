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

add_subdirectory(tests)
add_subdirectory(examples)