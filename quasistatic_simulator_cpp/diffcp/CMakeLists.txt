add_library(log_barrier_solver log_barrier_solver.cc)
target_link_libraries(log_barrier_solver drake::drake)
target_include_directories(log_barrier_solver PUBLIC ${PROJECT_SOURCE_DIR})

add_library(optimization_derivatives qp_derivatives.cc socp_derivatives.cc)
target_link_libraries(optimization_derivatives drake::drake)
target_include_directories(optimization_derivatives PUBLIC ${PROJECT_SOURCE_DIR})
