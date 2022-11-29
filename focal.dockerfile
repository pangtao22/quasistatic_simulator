FROM robotlocomotion/drake:focal

#COPY scripts/install_prereqs.sh /tmp/
COPY requirements.txt /tmp/requirements.txt
RUN #/tmp/install_prereqs.sh
RUN apt -y install xvfb git
#RUN python3 -m pip install -r /tmp/requirements.txt  # errors right now

# put drake on the python path.
#ENV PYTHONPATH /opt/drake/lib/python3.8/site-packages:$PYTHONPATH
