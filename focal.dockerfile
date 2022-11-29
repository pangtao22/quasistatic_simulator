FROM robotlocomotion/drake:focal

## install python requirements
COPY requirements.txt /requirementx.txt
RUN pip3 install -r requirements.txt

# put drake on the python path.
#ENV PYTHONPATH /opt/drake/lib/python3.8/site-packages:$PYTHONPATH
