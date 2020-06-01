FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel 
# Avoid warnings by switching to noninteractive

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git openssh-client iproute2 procps iproute2 lsb-release \
    #
    # TBC if all of this is required (from AIde scripts)
    && apt-get -y install libpq-dev python-dev wget systemd sudo

# TESTING STARTS HERE
RUN conda install -c conda-forge bottle tqdm bcrypt netifaces gunicorn requests \
 && conda install -c anaconda psycopg2

RUN sudo apt-get install -y software-properties-common

RUN pip install celery[redis,auth,msgpack]

COPY ./librabbitmq-2.0.0-cp37-cp37m-linux_x86_64.whl ./librabbitmq-2.0.0-cp37-cp37m-linux_x86_64.whl

ENTRYPOINT bash