FROM debian:buster-slim as base

FROM base as builder

MAINTAINER CO2MPAS <vinci1it2000@gmail.com>

# Add library to compile C-code.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-dev python3-pip gcc libyaml-dev && \
    pip3 install wheel setuptools --no-cache-dir

# Install CO2MPAS requirements.
COPY ./requirements /tmp/co2mpas/requirements
RUN cd /tmp/co2mpas/requirements && pip3 install -r all.pip --no-cache-dir

# Install CO2MPAS.
COPY ./co2mpas /tmp/co2mpas/co2mpas
COPY ./doc/conf.py /tmp/co2mpas/doc/conf.py
COPY ./setup.py /tmp/co2mpas/setup.py

RUN cd /tmp/co2mpas && \
    python3 setup.py bdist_wheel && \
    WHEEL=$(find /tmp/co2mpas/dist/co2mpas*.whl) && \
    pip3 install $WHEEL --no-dependencies --no-cache-dir

FROM base

COPY --from=builder /usr/local /usr/local

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 libyaml-0-2 libgomp1 python3-lib2to3 python3-pip graphviz && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.7 /usr/bin/python

RUN mkdir /data
WORKDIR /data
