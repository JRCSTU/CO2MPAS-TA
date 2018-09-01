## monorepo utilities for the developers
#
PNAME			:= co2mpas
SUBPROJECTS 	:= pCO2SIM pCO2DICE pCO2GUI

include Makefile.defs
BUILDALL		:= $(addsuffix -all,$(BUILDCMDS))

default2:
	@echo Specify one of: $(BUILDALL)
default: default2


wheel-all		: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
install-all		: _installwarn $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
uninstall-all	: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
clean-build-all	: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
clean-doc-all	: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)
clean-all		: $(SUBPROJECTS) $(MAKECMDGOALS:-all=)

## Install all projects in "develop" mode.
develop-all: 
	pip install $(addprefix -e ./,$(SUBPROJECTS)) -e .

twine-all: #wheel-all
	twine upload -su $(USER) $(addsuffix /dist/*.whl,$(SUBPROJECTS)) dist/*.whl


$(SUBPROJECTS):
	$(MAKE) -C $@ $(MAKECMDGOALS:-all=)


.PHONY: default $(BUILDCMDS) $(BUILDALL) $(SUBPROJECTS)
