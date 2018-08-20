## monorepo utilities for the developers
#
PNAME			:= co2mpas
SUBPROJECTS 	:= pCO2SIM pCO2DEPS pCO2DICE pCO2GUI

include Makefile.defs
BUILDALL		:= $(addsuffix -all,$(BUILDCMDS))


wheel-all		: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
install-all		: _installwarn $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
clean-build-all	: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
clean-doc-all	: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
clean-all		: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)

## Install all projects in "develop" mode.
develop-all: 
	pip install $(addprefix -e ./,$(SUBPROJECTS)) -e .

uninstall-all:
	pip uninstall -y $(SUBPROJECTS) $(PNAME)




$(SUBPROJECTS):
	$(MAKE) -C $@ $(MAKECMDGOALS:-all=)


.PHONY: default $(BUILDCMDS) $(BUILDALL) $(SUBPROJECTS)
