#!/usr/bin/env bash
set -euxo pipefail

cd $QSIM_CPP_PATH/build
ctest -V .

#cd /github/workspace
pytest .


