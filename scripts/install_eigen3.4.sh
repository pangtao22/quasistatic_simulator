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
