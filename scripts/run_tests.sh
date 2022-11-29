#!/usr/bin/env bash
set -euxo pipefail

# proof of life
python3 /github/workspace/qsim/tests/proof_of_life.py

# run tests
#ctest -V .
