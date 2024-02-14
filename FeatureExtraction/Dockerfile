# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image

FROM python:3.11

LABEL maintainer="Sam Border - Computational Microscopy Imaging Lab. <sumanth.devarasetty@medicine.ufl.edu> <samuel.border@medicine.ufl.edu>"

RUN apt-get update && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    software-properties-common \
    libssl-dev \
    # Standard build tools \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libmemcached-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update ##[edited]

RUN apt-get install libxml2-dev libxslt1-dev -y

# Required for opencv-python (cv2)
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /
# Make Python3 the default and install pip.  Whichever is done last determines
# the default python version for pip.

#
RUN which  python && \
    python --version

ENV build_path=$PWD/build
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copying location of feature extraction scripts
ENV ftx_path=$PWD/feature_ext
RUN mkdir -p $ftx_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . $ftx_path/
WORKDIR $ftx_path

#   Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.

# Installing packages in setup.py
RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    pip install --no-cache-dir .  && \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN python --version && pip --version && pip freeze

# define entrypoint through which all CLIs can be run
WORKDIR $ftx_path/feature_ext/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint CODEX_FTX --help


ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]