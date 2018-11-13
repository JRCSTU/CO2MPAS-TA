## monorepo utilities for the developers
#
PNAME			:= co2sim
SOURCES 	= $(shell find src -not -name '*cache*')

include ../Makefile.defs

DOCKING		:= $(addprefix docker/,docker-compose.yml Dockerfile)


docker: wheel $(DOCKING)
	docker-compose -f docker/docker-compose.yml build 