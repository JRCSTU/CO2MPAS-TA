# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to define a sub-model from a dsp.
"""


def define_sub_model(dsp, inputs, outputs, models):
    """
    Defines a sub-model from a dsp.

    :param dsp:
        Original model.
    :type dsp: schedula.Dispatcher

    :param inputs:
        Data inputs.
    :type inputs: list | tuple

    :param outputs:
        Data outputs.
    :type outputs: list | tuple

    :param models:
        Data models.
    :type models: list | tuple

    :return:
        A sub-model.
    :rtype: schedula.Dispatcher
    """
    assert not set(models).difference(dsp.nodes), models
    sub = dsp.shrink_dsp(set(inputs or []).union(models), outputs)
    assert set(sub.nodes).issuperset(set(inputs).union(outputs)), models
    return sub


def _accuracy_score(*args, **kwargs):
    from sklearn.metrics import accuracy_score as func
    return func(*args, **kwargs)
