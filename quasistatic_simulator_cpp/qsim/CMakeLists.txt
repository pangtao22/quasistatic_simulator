add_library(contact_computer contact_jacobian_calculator.h
        contact_jacobian_calculator.cc quasistatic_sim_params.h)
target_link_libraries(contact_computer drake::drake)
target_include_directories(contact_computer PUBLIC ${PROJECT_SOURCE_DIR})

add_library(get_model_paths get_model_paths.h get_model_paths.cc)
target_link_libraries(get_model_paths drake::drake)
target_include_directories(get_model_paths PUBLIC ${PROJECT_SOURCE_DIR})

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
target_include_directories(quasistatic_simulator PUBLIC ${PROJECT_SOURCE_DIR})
