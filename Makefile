## monorepo utilities for the developers
#
PNAME			:= co2mpas

SUBPROJECTS 	:= pCO2SIM pCO2DICE pCO2GUI
TOPTARGETS		:= wheels develop uninstall clean


wheel: $(SUBPROJECTS)
	python setup.py bdist_wheel

## Install all projects in "develop" mode.
develop: 
	pip install $(addprefix -e ./,$(SUBPROJECTS)) -e .

uninstall:
	pip uninstall -y $(SUBPROJECTS) $(PNAME)

clean: $(SUBPROJECTS)



$(SUBPROJECTS):
	$(MAKE) -C $@ $(MAKECMDGOALS)


.PHONY: default $(TOPTARGETS) $(SUBPROJECTS)
