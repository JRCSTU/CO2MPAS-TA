#!/usr/bin/env bash
ver=$(python ../co2mpas/_version.py --version)
export CO2MPAS_TAG=vinci1it2000/co2mpas:v${ver}
echo "Building image $CO2MPAS_TAG..."
docker-compose build
