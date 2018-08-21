FROM debian:latest

#  $ docker build . -t continuumio/miniconda3:latest -t continuumio/miniconda3:4.5.4
#  $ docker run --rm -it continuumio/miniconda3:latest /bin/bash
#  $ docker push continuumio/miniconda3:latest
#  $ docker push continuumio/miniconda3:4.5.4

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

## Install requirements.
#
COPY ./requirements /tmp/co2mpas/
RUN cd /tmp/co2mpas/ && \
    bash ./install_conda_reqs.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    rm -rf /tmp/co2mpas

## Install CO2MPAS.
#
COPY ./dist/co2sim-*.whl /tmp/co2mpas/
RUN WHEEL="$(find /tmp/co2mpas/co2sim-*.whl)" && \
    pip install --no-cache-dir $WHEEL[io] && \
    rm -rf /tmp/co2mpas

CMD co2mpas batch /data/input -O /data/output -D flag.engineering_mode=True
