#!/usr/bin/env bash
set -euxo pipefail

# install Eigen 3.4
curl -o eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen.tar.gz -C /tmp
rm eigen.tar.gz
cd /tmp/eigen-3.4.0
mkdir build
cd build
cmake ..
make install
rm -rf /tmp/eigen-3.4.0

# build quasistatic_simulator_cpp
cd /quasistatic_simulator_cpp
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/drake -DEigen3_DIR:PATH=/usr/local/lib/cmake/eigen3 -DCMAKE_BUILD_TYPE=Debug ..
make
