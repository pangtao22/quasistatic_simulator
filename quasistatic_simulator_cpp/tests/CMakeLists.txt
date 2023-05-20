add_library(test_utilities test_utilities.h test_utilites.cc)
target_link_libraries(test_utilities drake::drake)
target_include_directories(test_utilities PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_batch_simulator test_batch_simulator.cc)
target_link_libraries(test_batch_simulator quasistatic_simulator
        GTest::gtest test_utilities)
target_include_directories(test_batch_simulator PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_log_barrier_solver test_log_barrier_solver.cc)
target_link_libraries(test_log_barrier_solver
        log_barrier_solver GTest::gtest test_utilities)
target_include_directories(test_log_barrier_solver PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_contact_forces test_contact_forces.cc)
target_link_libraries(test_contact_forces
        quasistatic_simulator GTest::gtest test_utilities)
target_include_directories(test_contact_forces PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(test_quasistatic_sim_gradients test_quasistatic_sim_gradients.cc)
target_link_libraries(test_quasistatic_sim_gradients
        quasistatic_simulator GTest::gtest test_utilities)
target_include_directories(test_quasistatic_sim_gradients
        PUBLIC ${PROJECT_SOURCE_DIR})

add_test(NAME test_batch_simulator COMMAND test_batch_simulator)
add_test(NAME test_log_barrier_solver COMMAND test_log_barrier_solver)
add_test(NAME test_contact_forces COMMAND test_contact_forces)
add_test(NAME test_quasistatic_sim_gradients
        COMMAND test_quasistatic_sim_gradients)
