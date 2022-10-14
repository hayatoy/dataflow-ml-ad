FROM nvcr.io/nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

WORKDIR /pipeline

COPY requirements.txt .
COPY *.py ./

# If you need a different Python version, consider:
#   https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl g++ python3.9-dev python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10 \
    && curl https://bootstrap.pypa.io/get-pip.py | python \
    # Install the pipeline requirements and check that there are no conflicts.
    # Since the image already has all the dependencies installed,
    # there's no need to run with the --requirements_file option.
    && pip install --no-cache-dir -r requirements.txt \
    && pip check

# Set the entrypoint to Apache Beam SDK worker launcher.
COPY --from=apache/beam_python3.9_sdk:2.42.0rc1 /opt/apache/beam /opt/apache/beam
ENTRYPOINT [ "/opt/apache/beam/boot" ]