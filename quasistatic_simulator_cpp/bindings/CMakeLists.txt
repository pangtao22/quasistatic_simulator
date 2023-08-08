pybind11_add_module(qsim_cpp MODULE qsim_cpp.cc)
target_link_libraries(qsim_cpp PUBLIC quasistatic_simulator)
set_target_properties(qsim_cpp PROPERTIES CXX_VISIBILITY_PRESET default)

install(TARGETS qsim_cpp
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages"
)
