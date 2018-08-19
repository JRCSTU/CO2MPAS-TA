## monorepo utilities for the developers
#
PNAME			:= co2mpas

include Makefile.defs
SUBPROJECTS 	:= pCO2SIM pCO2DICE pCO2GUI


wheel: $(SUBPROJECTS)

## Install all projects in "develop" mode.
develop: $(SUBPROJECTS)

uninstall: $(SUBPROJECTS)

clean: $(SUBPROJECTS)



$(SUBPROJECTS):
	$(MAKE) -C $@ $(MAKECMDGOALS)


.PHONY: default $(TOPTARGETS) $(SUBPROJECTS)
