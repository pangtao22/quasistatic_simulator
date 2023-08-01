add_library(test_utilities test_utilities.h test_utilites.cc)
target_link_libraries(test_utilities drake::drake)
target_include_directories(test_utilities PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_batch_simulator batch_simulator_test.cc)
target_link_libraries(test_batch_simulator quasistatic_simulator
        GTest::gtest test_utilities)
target_include_directories(test_batch_simulator PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_log_barrier_solver log_barrier_solver_test.cc)
target_link_libraries(test_log_barrier_solver
        log_barrier_solver GTest::gtest test_utilities)
target_include_directories(test_log_barrier_solver PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_contact_forces contact_forces_test.cc)
target_link_libraries(test_contact_forces
        quasistatic_simulator GTest::gtest test_utilities)
target_include_directories(test_contact_forces PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_quasistatic_sim_gradients quasistatic_sim_gradients_test.cc)
target_link_libraries(test_quasistatic_sim_gradients
        quasistatic_simulator GTest::gtest test_utilities)
target_include_directories(test_quasistatic_sim_gradients
        PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_solver_selector solver_selector_test.cc)
target_link_libraries(test_solver_selector GTest::gtest log_barrier_solver)
target_include_directories(test_solver_selector PUBLIC ${PROJECT_SOURCE_DIR})

add_test(NAME test_batch_simulator COMMAND test_batch_simulator)
add_test(NAME test_log_barrier_solver COMMAND test_log_barrier_solver)
add_test(NAME test_contact_forces COMMAND test_contact_forces)
add_test(NAME test_quasistatic_sim_gradients
        COMMAND test_quasistatic_sim_gradients)
add_test(NAME test_solver_selector COMMAND test_solver_selector)
