#!/usr/bin/env bash
set -euxo pipefail

# build quasistatic_simulator_cpp
cd /github/workspace/quasistatic_simulator_cpp
mkdir cmake_build_release && cd cmake_build_release
cmake -DCMAKE_PREFIX_PATH=/opt/drake -DCMAKE_BUILD_TYPE=Debug ..
make -j

# run tests
#ctest -V .
