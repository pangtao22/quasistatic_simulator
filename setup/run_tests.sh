#!/usr/bin/env bash
set -euxo pipefail

#cd /github/workspace
pwd
pytest .

pushd $QSIM_CPP_PATH/build
ctest -V .
popd


