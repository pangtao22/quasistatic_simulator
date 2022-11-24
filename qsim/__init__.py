# empty.
import sys
from pathlib import Path, PurePath

# Assuming that the python bindings are built into root/cmake_build_release/src
qsim_cpp_path = PurePath(
    Path(__file__).parent,
    "..",
    "quasistatic_simulator_cpp",
    "cmake_build_release",
    "src",
)
sys.path.append(str(qsim_cpp_path))
