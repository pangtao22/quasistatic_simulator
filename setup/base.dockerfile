FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
EXPOSE 7000-7099/tcp
EXPOSE 8888/tcp

RUN apt-get update \
      && apt-get install --no-install-recommends -qy curl apt-transport-https \
      sudo ca-certificates libgtest-dev libgflags-dev python3.10-dev  \
      python3-pip git python-is-python3 libyaml-cpp-dev \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean all

ENV DRAKE_URL=https://github.com/RobotLocomotion/drake/releases/download/v1.15.0/drake-20230418-jammy.tar.gz
RUN curl -fSL -o drake.tar.gz $DRAKE_URL
RUN tar -xzf drake.tar.gz -C /opt && rm drake.tar.gz
RUN apt-get update \
  && yes "Y" | bash /opt/drake/share/drake/setup/install_prereqs \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean all

# Install Eigen 3.4
COPY setup/install_eigen3.4.sh /tmp/
RUN /tmp/install_eigen3.4.sh

# Install additional python dependencies
COPY ./setup/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

## Build QuasistaticSimulatorCpp and its python bindings.
ENV QSIM_PATH /quasistatic_simulator
ENV QSIM_CPP_PATH $QSIM_PATH/quasistatic_simulator_cpp

# put qsim_cpp on the python path.
ENV PYTHONPATH $QSIM_CPP_PATH/build/bindings:$PYTHONPATH
ENV PYTHONPATH $QSIM_PATH:$PYTHONPATH
ENV PYTHONPATH /opt/drake/lib/python3.10/site-packages:$PYTHONPATH
