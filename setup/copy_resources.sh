#!/usr/bin/env bash
set -euxo pipefail

# copy stuff
# TODO: install instead of copy.
mkdir $QSIM_PATH
cp -r models $QSIM_PATH
cp -r robotics_utilities $QSIM_PATH
cp -r qsim $QSIM_PATH
cp -r quasistatic_simulator_cpp $QSIM_CPP_PATH
