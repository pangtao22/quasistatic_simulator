FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && yes "Y" \
      | apt-get install --no-install-recommends curl apt-transport-https sudo \
      ca-certificates libgtest-dev libgflags-dev python3.8-dev git \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean all

ENV DRAKE_URL=https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-focal.tar.gz
RUN curl -o drake.tar.gz $DRAKE_URL
RUN tar -xzf drake.tar.gz -C /opt && rm drake.tar.gz
RUN apt-get update \
  && yes "Y" | bash /opt/drake/share/drake/setup/install_prereqs \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean all

# Install Eigen 3.4
COPY scripts/install_eigen3.4.sh /tmp/
RUN /tmp/install_eigen3.4.sh

# Install additional python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt  # errors right now

# Build QuasistaticSimulatorCpp and its python bindings.
ENV QSIM_PATH /quasistatic_simulator
ENV QSIM_CPP_PATH $QSIM_PATH/quasistatic_simulator_cpp
COPY scripts/build_bindings.sh /tmp/
COPY models $QSIM_PATH/models/
COPY robotics_utilities $QSIM_PATH/robotics_utilities
COPY quasistatic_simulator_cpp/ $QSIM_CPP_PATH/
RUN /tmp/build_bindings.sh

# put qsim_cpp on the python path.
ENV PYTHONPATH $QSIM_CPP_PATH/build/src:$PYTHONPATH
