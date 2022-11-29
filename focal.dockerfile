FROM robotlocomotion/drake:focal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && yes "Y" \
      | apt-get install --no-install-recommends curl apt-transport-https sudo \
      ca-certificates libgtest-dev libgflags-dev python3.8-dev\
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean all

RUN apt-get update \
  && yes "Y" | bash /opt/drake/share/drake/setup/install_prereqs \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean all

#COPY scripts/install_prereqs.sh /tmp/
COPY requirements.txt /tmp/requirements.txt
#RUN python3 -m pip install -r /tmp/requirements.txt  # errors right now

# put drake on the python path.
#ENV PYTHONPATH /opt/drake/lib/python3.8/site-packages:$PYTHONPATH
