add_library(contact_computer contact_jacobian_calculator.h
        contact_jacobian_calculator.cc quasistatic_sim_params.h)
<<<<<<< Updated upstream
target_link_libraries(contact_computer drake::drake)
target_include_directories(contact_computer PUBLIC ${PROJECT_SOURCE_DIR})

add_library(get_model_paths get_model_paths.cc)
target_link_libraries(get_model_paths drake::drake)
target_include_directories(get_model_paths PUBLIC ${PROJECT_SOURCE_DIR})

add_library(get_model_paths get_model_paths.cc)
target_link_libraries(get_model_paths PUBLIC drake::drake)
>>>>>>> Stashed changes
>>>>>>> Stashed changes

add_library(quasistatic_simulator
        quasistatic_simulator.cc
        batch_quasistatic_simulator.cc
        quasistatic_parser.cc
        finite_differencing_gradient.cc)
<<<<<<< Updated upstream
target_link_libraries(quasistatic_simulator optimization_derivatives
        drake::drake get_model_paths contact_computer log_barrier_solver
        yaml-cpp)
target_include_directories(quasistatic_simulator PUBLIC ${PROJECT_SOURCE_DIR})
