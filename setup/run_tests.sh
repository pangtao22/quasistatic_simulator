#!/usr/bin/env bash
set -euxo pipefail

cd /github/workspace
pytest .

cd $QSIM_CPP_PATH/build
ctest -V .
