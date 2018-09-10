FROM continuumio/miniconda3:4.5.4

# Install requirements.
COPY ./requirements/install_conda_reqs.sh /tmp/co2mpas/requirements/install_conda_reqs.sh
RUN cd /tmp/co2mpas/requirements && \
    bash install_conda_reqs.sh

COPY ./requirements/exe.pip /tmp/co2mpas/requirements/exe.pip
RUN cd /tmp/co2mpas/requirements && \
    pip install -r exe.pip --no-cache-dir

# Install CO2MPAS.
COPY ./src/co2mpas /tmp/co2mpas/src/co2mpas
COPY ./setup.py /tmp/co2mpas/setup.py
COPY ./README.rst /tmp/co2mpas/README.rst

ARG co2sim_VERSION
ENV co2sim_VERSION "$co2sim_VERSION"
RUN cd /tmp/co2mpas && \
    python setup.py bdist_wheel && \
    WHEEL=$(find /tmp/co2mpas/dist/co2sim*.whl) && \
    pip install $WHEEL[io] --no-cache-dir && \
    rm -rf /tmp/co2mpas /opt/conda/pkgs/cache

CMD co2mpas batch /data/input -O /data/output -D flag.engineering_mode=True
