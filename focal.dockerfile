FROM robotlocomotion/drake:focal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && yes "Y" \
      | apt-get install --no-install-recommends curl apt-transport-https sudo \
      ca-certificates libgtest-dev libgflags-dev python3.8-dev git \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean all

RUN apt-get update \
  && yes "Y" | bash /opt/drake/share/drake/setup/install_prereqs \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean all

# Build QuasistaticSimulatorCpp and its python bindings.
COPY scripts/build_bindings.sh /tmp/
COPY quasistatic_simulator_cpp/ /quasistatic_simulator_cpp/
RUN /tmp/build_bindings.sh

# Install additional python dependencies
COPY requirements.txt /tmp/requirements.txt
#RUN python3 -m pip install -r /tmp/requirements.txt  # errors right now

# put qsim_cpp on the python path.
ENV PYTHONPATH /quasistatic_simulator_cpp/build/src:$PYTHONPATH
