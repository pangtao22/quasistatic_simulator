#!/usr/bin/env bash
set -euxo pipefail

# proof of life
cd /github/workspace
pytest .

# run C++ tests
cd $QSIM_CPP_PATH/build
ctest -V .
