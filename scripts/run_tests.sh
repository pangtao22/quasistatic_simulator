#!/usr/bin/env bash
set -euxo pipefail

# proof of life
python3 /github/workspace/qsim/tests/proof_of_life.py

# run C++ tests
ctest -V $QSIM_CPP_PATH/build
