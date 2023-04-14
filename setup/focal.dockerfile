FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
EXPOSE 7000-7099/tcp
EXPOSE 8888/tcp

RUN apt-get update \
      && apt-get install --no-install-recommends -qy curl apt-transport-https \
      sudo ca-certificates libgtest-dev libgflags-dev python3.8-dev  \
      python3-pip git python-is-python3 \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean all

ENV DRAKE_URL=https://github.com/RobotLocomotion/drake/releases/download/v1.11.0/drake-20221214-focal.tar.gz
RUN curl -fSL -o drake.tar.gz $DRAKE_URL
RUN tar -xzf drake.tar.gz -C /opt && rm drake.tar.gz
RUN apt-get update \
  && yes "Y" | bash /opt/drake/share/drake/setup/install_prereqs \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean all

# Install Eigen 3.4
COPY install_eigen3.4.sh /tmp/
RUN /tmp/install_eigen3.4.sh

# Install additional python dependencies
COPY ../requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

## Build QuasistaticSimulatorCpp and its python bindings.
#ENV QSIM_PATH /quasistatic_simulator
#ENV QSIM_CPP_PATH $QSIM_PATH/quasistatic_simulator_cpp
#COPY build_bindings.sh /tmp/
#COPY ../models $QSIM_PATH/models/
#COPY ../robotics_utilities $QSIM_PATH/robotics_utilities
#COPY ../qsim $QSIM_PATH/qsim
#COPY ../quasistatic_simulator_cpp $QSIM_CPP_PATH/
#RUN /tmp/build_bindings.sh
#
## put qsim_cpp on the python path.
#ENV PYTHONPATH $QSIM_CPP_PATH/build/src:$PYTHONPATH
#ENV PYTHONPATH $QSIM_PATH:$PYTHONPATH
#ENV PYTHONPATH /opt/drake/lib/python3.8/site-packages:$PYTHONPATH
