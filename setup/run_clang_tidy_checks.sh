#!/usr/bin/env bash
set -euxo pipefail

# build quasistatic_simulator_cpp
cd $QSIM_CPP_PATH
mkdir build-tidy && cd build-tidy
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake \
      -DCMAKE_PREFIX_PATH=/opt/drake \
      -DEigen3_DIR:PATH=/usr/local/lib/cmake/eigen3 \
      -DCMAKE_BUILD_TYPE=Debug \
      ..

run_clang_tidy -quiet -style=../.clang-tidy \
  '^.*/(bindings|diffcp|qsim)/.+\.(h|cc)$'
