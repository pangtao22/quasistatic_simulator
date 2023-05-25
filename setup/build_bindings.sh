#!/usr/bin/env bash
set -euxo pipefail

# build quasistatic_simulator_cpp
cd $QSIM_CPP_PATH
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake -DCMAKE_PREFIX_PATH=/opt/drake -DEigen3_DIR:PATH=/usr/local/lib/cmake/eigen3 -DCMAKE_BUILD_TYPE=Debug ..
make -j 2
