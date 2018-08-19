## monorepo utilities for the developers
#
PNAME			:= co2mpas
SUBPROJECTS 	:= pCO2SIM pCO2DICE pCO2GUI

include Makefile.defs
BUILDALL		:= wheel-all develop-all uninstall-all clean-all


wheel-all: $(SUBPROJECTS) wheel

## Install all projects in "develop" mode.
develop-all: 
	pip install $(addprefix -e ./,$(SUBPROJECTS)) -e .

uninstall-all:
	pip uninstall -y $(SUBPROJECTS) $(PNAME)

clean-all: $(SUBPROJECTS)



$(SUBPROJECTS):
	$(MAKE) -C $@ $(MAKECMDGOALS:-all=)


.PHONY: default $(BUILDCMDS) $(BUILDALL) $(SUBPROJECTS)
