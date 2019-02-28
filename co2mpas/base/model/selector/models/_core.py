# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains functions to compare/select the CO2MPAS calibrated models.

Docstrings should provide sufficient understanding for any individual function.

Modules:

.. currentmodule:: co2mpas.model.selector

.. autosummary::
    :nosignatures:
    :toctree: selector/

    co2_params
"""


def define_sub_model(dsp, inputs, outputs, models):
    import schedula as sh
    if isinstance(dsp, sh.Blueprint):
        dsp = dsp.register()
    missing = set(outputs).difference(dsp.nodes)
    if missing:
        outputs = set(outputs).difference(missing)
    if inputs is not None:
        inputs = set(inputs).union(models)
    return dsp.shrink_dsp(inputs, outputs)
