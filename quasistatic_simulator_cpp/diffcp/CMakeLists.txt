add_library(log_barrier_solver log_barrier_solver.cc solver_selector.cc)
target_link_libraries(log_barrier_solver PUBLIC drake::drake)
target_include_directories(log_barrier_solver
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

add_library(optimization_derivatives qp_derivatives.cc socp_derivatives.cc)
target_link_libraries(optimization_derivatives PUBLIC drake::drake)
target_include_directories(optimization_derivatives
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

install(TARGETS log_barrier_solver optimization_derivatives
        EXPORT ${CMAKE_PROJECT_NAME}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
