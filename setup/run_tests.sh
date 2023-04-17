#!/usr/bin/env bash
set -euxo pipefail

pytest .

cd $QSIM_CPP_PATH/build
ctest -V .
