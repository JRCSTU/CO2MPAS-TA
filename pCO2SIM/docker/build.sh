#!/usr/bin/env bash

ver=$(python ../src/co2mpas/__init__.py version)
export co2mpas_VERSION=${ver//[+]/-}
export CO2MPAS_TAG=vinci1it2000/co2mpas:v${ver//[+]/-}
echo "Building image $CO2MPAS_TAG..."
docker-compose build
