#!/usr/bin/env bash
set -euxo pipefail

# copy stuff
cp setup/build_bindings.sh /tmp/
cp -r models $QSIM_PATH
cp -r robotics_utilities $QSIM_PATH
cp -r qsim $QSIM_PATH
cp -r quasistatic_simulator_cpp $QSIM_CPP_PATH

# build quasistatic_simulator_cpp
cd $QSIM_CPP_PATH
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/drake -DEigen3_DIR:PATH=/usr/local/lib/cmake/eigen3 -DCMAKE_BUILD_TYPE=Debug ..
make -j 2
