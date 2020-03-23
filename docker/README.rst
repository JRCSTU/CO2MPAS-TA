CO2MPAS docker images
=====================
There are two images available:
 - co2mpas-debian
 - co2mpas-exe

Quick Start
-----------
You may find usage Guidelines in the wiki:
https://github.com/JRCSTU/CO2MPAS-TA/wiki/CO2MPAS-user-guideline

To create work folders and then fill input with sample-vehicles:

`$ md input output`
`$ docker run -it --rm -v $(pwd):/data vinci1it2000/co2mpas-debian:latest co2mpas demo /data/input`

To run CO2MPAS with batch cmd:

`$ docker run -it --rm -v $(pwd):/data vinci1it2000/co2mpas-debian:latest co2mpas run /data/input -O /data/output`

You can run a specific version changing tag `latest` with the desired version:

`$ docker run -it --rm -v $(pwd):/data vinci1it2000/co2mpas-debian:vX.X.X co2mpas run /data/input -O /data/output`

Build docker images
-------------------
To build locally all images as `latest`, you can run the following command:

`$ docker-compose build`

if you want to build just one image:

`$ docker-compose build co2mpas-debian`

If you want to build and tag all images with the current version, you can run
the following command:

`$ bash ./build.sh`

if you want to build just one image:

`$ bash ./build.sh co2mpas-debian`