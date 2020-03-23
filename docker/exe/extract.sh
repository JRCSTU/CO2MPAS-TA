#!/usr/bin/env bash
cd "$(dirname "$0")"
docker create -ti --name dummy vinci1it2000/co2mpas-exe:v4.1.10 bash
docker cp dummy:/dist ./dist
docker rm -f dummy
zip -r ./dist.zip ./dist
