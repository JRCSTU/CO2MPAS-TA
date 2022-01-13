# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to define a sub-model from a dsp.
"""

_cache_sub_model = {}


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
    sub_id = dsp.name, tuple(inputs), tuple(outputs), tuple(models)
    if sub_id in _cache_sub_model:
        return _cache_sub_model[sub_id].copy()
    assert not set(models).difference(dsp.nodes), models
    sub = dsp.shrink_dsp(set(inputs or []).union(models), outputs)
    assert set(sub.nodes).issuperset(set(inputs).union(outputs)), models
    _cache_sub_model[sub_id] = sub
    return sub


def _accuracy_score(*args, **kwargs):
    from sklearn.metrics import accuracy_score as func
    return func(*args, **kwargs)
