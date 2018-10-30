CO2MPAS docker images
=====================
There are two images available:
 - co2mpas-alpine
 - co2mpas-debian

Quick Start
-----------
You may find usage Guidelines in the wiki:
https://github.com/JRCSTU/CO2MPAS-TA/wiki/CO2MPAS-user-guideline

To create work folders and then fill input with sample-vehicles:

`$ md input output`
`$ docker run -it --rm -v $(pwd):/data vinci1it2000/co2mpas-alpine:latest co2mpas demo /data/input`

To run CO2MPAS with batch cmd:

`$ docker run -it --rm -v $(pwd):/data vinci1it2000/co2mpas-alpine:latest co2mpas batch /data/input -O /data/output -D flag.engineering_mode=True`

You can run a specific version changing tag `latest` with the desired version:

`$ docker run -it --rm -v $(pwd):/data vinci1it2000/co2mpas-alpine:2.0.0 co2mpas batch /data/input -O /data/output -D flag.engineering_mode=True`

Build docker images
-------------------
To build locally all images as `latest`, you can run the following command:

`$ docker-compose build`

if you want to build just one image:

`$ docker-compose build co2mpas-alpine`

If you want to build and tag all images with the current version, you can run
the following command:

`$ bash ./build.sh`

if you want to build just one image:

`$ bash ./build.sh co2mpas-alpine`