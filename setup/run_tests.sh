#!/usr/bin/env bash
set -euxo pipefail

pwd  # FYI
pytest .

pushd $QSIM_CPP_PATH/build
ctest -V .
popd
