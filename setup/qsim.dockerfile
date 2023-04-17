FROM ghcr.io/pangtao22/quasistatic_simulator_base:main

# TODO: replace copy with instllation.
COPY models $QSIM_PATH/models
COPY robotics_utilities $QSIM_PATH/robotics_utilities
COPY qsim $QSIM_PATH/qsim
COPY quasistatic_simulator_cpp $QSIM_CPP_PATH

COPY ./setup/build_bindings.sh /tmp/
RUN /tmp/build_bindings.sh
