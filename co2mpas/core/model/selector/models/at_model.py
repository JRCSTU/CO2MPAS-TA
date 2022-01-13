# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and constants to define the at_model selector.
"""
import logging
import collections
import schedula as sh
import co2mpas.utils as co2_utl
from ._core import define_sub_model, _accuracy_score
from ...physical.gear_box.at_gear import dsp as _at_gear
from ...physical.gear_box.mechanical import calculate_gear_box_speeds_in

log = logging.getLogger(__name__)

#: Model name.
name = 'at_model'

#: Parameters that constitute the model.
models = [
    'MVL', 'CMV', 'CMV_Cold_Hot', 'DTGS', 'GSPV', 'GSPV_Cold_Hot',
    'specific_gear_shifting', 'change_gear_window_width',
    'max_velocity_full_load_correction', 'plateau_acceleration'
]

#: Inputs required to run the model.
inputs = [
    'idle_engine_speed', 'full_load_curve', 'accelerations', 'motive_powers',
    'engine_speeds_out', 'engine_temperatures', 'plateau_acceleration',
    'time_cold_hot_transition', 'times', 'stop_velocity', 'cycle_type',
    'use_dt_gear_shifting', 'specific_gear_shifting', 'velocity_speed_ratios',
    'velocities', 'MVL', 'fuel_saving_at_strategy', 'change_gear_window_width',
    'max_velocity_full_load_correction'
]

#: Relevant outputs of the model.
outputs = ['gears']

#: Targets to compare the outputs of the model.
targets = outputs

#: Weights coefficients to compute the model score.
weights = sh.map_list(targets, -1)

#: Metrics to compare outputs with targets.
metrics = {'gears': _accuracy_score}


def _correlation_coefficient(t, o):
    import numpy as np
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.corrcoef(t, o)[0, 1] if t.size > 1 else np.nan


def calculate_error_coefficients(
        identified_gears, gears, engine_speeds, predicted_engine_speeds,
        velocities, stop_velocity):
    """
    Calculates the prediction's error coefficients.

    :param identified_gears:
        Identified gear vector [-].
    :type identified_gears: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param engine_speeds:
        Engine speed vector [RPM].
    :type engine_speeds: numpy.array

    :param predicted_engine_speeds:
        Predicted Engine speed vector [RPM].
    :type predicted_engine_speeds: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Correlation coefficient and mean absolute error.
    :rtype: dict
    """

    b = velocities > stop_velocity

    x = engine_speeds[b]
    y = predicted_engine_speeds[b]
    res = {
        'mean_absolute_error': co2_utl.mae(x, y),
        'correlation_coefficient': _correlation_coefficient(x, y),
        'accuracy_score': _accuracy_score(identified_gears, gears)
    }

    return res


def select_models(keys, data):
    """
    Select models from data.

    :param keys:
        Model keys.
    :type keys: list

    :param data:
        Cycle data.
    :type data: dict

    :return:
        Models.
    :rtype: dict
    """
    sgs = 'specific_gear_shifting'
    # Namespace shortcuts.
    try:
        vel, vsr = data['velocities'], data['velocity_speed_ratios']
        t_eng, t_gears = data['engine_speeds_out'], data['gears']
        sv, at_m = data['stop_velocity'], data[sgs]
    except KeyError:
        return {}

    t_e = 'mean_absolute_error', 'accuracy_score', 'correlation_coefficient'

    # at_models to be assessed.
    if at_m == 'ALL':
        at_m = {'CMV', 'CMV_Cold_Hot', 'DTGS', 'GSPV', 'GSPV_Cold_Hot'}
    else:
        at_m = {at_m}

    # Other models to be taken from calibration output.
    mdl = sh.selector(set(keys) - at_m, data, allow_miss=True)

    # Inputs to predict the gears.
    inp = sh.selector(inputs, data, allow_miss=True)

    func = _at_gear.register()

    def _err(model_id, model):
        gears = func(
            inputs=sh.combine_dicts(inp, {sgs: model_id, model_id: model}),
            outputs=['gears']
        )['gears']

        eng = calculate_gear_box_speeds_in(gears, vel, vsr, sv)
        return calculate_error_coefficients(t_gears, gears, t_eng, eng, vel, sv)

    def _sort(v):
        e = sh.selector(t_e, v[0], output_type='list')
        return (e[0], -e[1], -e[2]), v[1]

    # Sort by error.
    rank = sorted((
        (_err(k, m), k, m)
        for k, m in sorted(sh.selector(at_m, data, allow_miss=True).items())
    ), key=_sort)

    if rank:
        data['at_scores'] = collections.OrderedDict(
            (k, e) for e, k, m in rank)
        e, k, m = rank[0]
        mdl[sgs], mdl[k] = k, m
        log.debug(
            'at_gear_shifting_model: %s with mean_absolute_error %.3f '
            '[RPM], accuracy_score %.3f, and correlation_coefficient '
            '%.3f.', k, *sh.selector(t_e, e, output_type='list'))

    return mdl


#: Prediction model.
# noinspection PyProtectedMember
dsp = sh.Blueprint(_at_gear, inputs, outputs, models)._set_cls(
    define_sub_model
)
