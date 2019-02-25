FROM alpine:3.7

ENV LANG=C.UTF-8

# Here we install GNU libc (aka glibc) and set C.UTF-8 locale as default.

RUN ALPINE_GLIBC_BASE_URL="https://github.com/sgerrand/alpine-pkg-glibc/releases/download" && \
    ALPINE_GLIBC_PACKAGE_VERSION="2.28-r0" && \
    ALPINE_GLIBC_BASE_PACKAGE_FILENAME="glibc-$ALPINE_GLIBC_PACKAGE_VERSION.apk" && \
    ALPINE_GLIBC_BIN_PACKAGE_FILENAME="glibc-bin-$ALPINE_GLIBC_PACKAGE_VERSION.apk" && \
    ALPINE_GLIBC_I18N_PACKAGE_FILENAME="glibc-i18n-$ALPINE_GLIBC_PACKAGE_VERSION.apk" && \
    apk add --no-cache --virtual=.build-dependencies wget ca-certificates && \
    wget \
        "https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub" \
        -O "/etc/apk/keys/sgerrand.rsa.pub" && \
    wget \
        "$ALPINE_GLIBC_BASE_URL/$ALPINE_GLIBC_PACKAGE_VERSION/$ALPINE_GLIBC_BASE_PACKAGE_FILENAME" \
        "$ALPINE_GLIBC_BASE_URL/$ALPINE_GLIBC_PACKAGE_VERSION/$ALPINE_GLIBC_BIN_PACKAGE_FILENAME" \
        "$ALPINE_GLIBC_BASE_URL/$ALPINE_GLIBC_PACKAGE_VERSION/$ALPINE_GLIBC_I18N_PACKAGE_FILENAME" && \
    apk add --no-cache \
        "$ALPINE_GLIBC_BASE_PACKAGE_FILENAME" \
        "$ALPINE_GLIBC_BIN_PACKAGE_FILENAME" \
        "$ALPINE_GLIBC_I18N_PACKAGE_FILENAME" && \
    \
    rm "/etc/apk/keys/sgerrand.rsa.pub" && \
    /usr/glibc-compat/bin/localedef --force --inputfile POSIX --charmap UTF-8 C.UTF-8 || true && \
    echo "export LANG=C.UTF-8" > /etc/profile.d/locale.sh && \
    \
    apk del glibc-i18n && \
    \
    rm "/root/.wget-hsts" && \
    apk del .build-dependencies && \
    rm \
        "$ALPINE_GLIBC_BASE_PACKAGE_FILENAME" \
        "$ALPINE_GLIBC_BIN_PACKAGE_FILENAME" \
        "$ALPINE_GLIBC_I18N_PACKAGE_FILENAME"



ENV CONDA_DIR="/opt/conda"
ENV PATH="$CONDA_DIR/bin:$PATH"

# Install conda
RUN CONDA_VERSION="4.5.4" && \
    CONDA_MD5_CHECKSUM="a946ea1d0c4a642ddf0c3a26a18bb16d" && \
    \
    apk add --no-cache --virtual=.build-dependencies wget ca-certificates bash && \
    \
    mkdir -p "$CONDA_DIR" && \
    wget "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh" -O miniconda.sh && \
    echo "$CONDA_MD5_CHECKSUM  miniconda.sh" | md5sum -c && \
    bash miniconda.sh -f -b -p "$CONDA_DIR" && \
    echo "export PATH=$CONDA_DIR/bin:\$PATH" > /etc/profile.d/conda.sh && \
    rm miniconda.sh && \
    \
    conda update --all --yes && \
    conda config --set auto_update_conda False && \
    rm -r "$CONDA_DIR/pkgs/" && \
    \
    apk del --purge .build-dependencies && \
    \
    mkdir -p "$CONDA_DIR/locks" && \
chmod 777 "$CONDA_DIR/locks"

MAINTAINER CO2MPAS <vinci1it2000@gmail.com>

# Add library to compile C-code.
RUN apk --update --no-cache --virtual=.build-dependencies add gcc libc-dev git bash

# Add tk dependencies
RUN apk --update --no-cache add tk-dev

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
    pip install $WHEEL[io] --no-cache-dir

# Clean up.
RUN apk del --purge .build-dependencies && \
    conda clean --all && \
    rm -rf /tmp/co2mpas /opt/conda/pkgs/cache

CMD co2mpas batch /data/input -O /data/output -D flag.engineering_mode=True
