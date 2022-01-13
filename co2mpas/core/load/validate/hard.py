#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It provides functions to perform the hard validation.
"""
import logging
import functools
import numpy as np
import schedula as sh
from co2mpas.defaults import dfl

log = logging.getLogger(__name__)


def check_sign_currents(battery_currents, alternator_currents):
    """
    Checks if battery currents and alternator currents have the right signs.

    :param battery_currents:
        Low voltage battery current vector [A].
    :type battery_currents: numpy.array

    :param alternator_currents:
        Alternator currents [A].
    :type alternator_currents: numpy.array

    :return:
        If battery and alternator currents have the right signs.
    :rtype: (bool, bool)
    """
    #: Maximum allowed positive current for the alternator currents check [A].
    import co2mpas.utils as co2_utl
    max_pos_curr = 1.0
    b_c, a_c = battery_currents, alternator_currents

    a = co2_utl.reject_outliers(a_c, med=np.mean)[0]
    a = a <= max_pos_curr
    c = np.cov(a_c, b_c)[0][1]

    if c < 0:
        x = (a, a)
    elif c == 0:
        if any(b_c):
            x = (co2_utl.reject_outliers(b_c, med=np.mean)[0] <= 0, a)
        else:
            x = (True, a)
    else:
        x = (not a, a)
    return x


# noinspection PyUnusedLocal
def _check_sign_currents(data, *args):
    c = ('service_battery_currents', 'alternator_currents')
    try:
        a = sh.selector(c, data, output_type='list')
        s = check_sign_currents(*a)
        if not all(s):
            s = ' and '.join([k for k, v in zip(c, s) if not v])
            msg = "Probably '{}' have/has the wrong sign!".format(s)
            return c, msg
    except KeyError:  # `c` is not in `data`.
        pass


# noinspection PyUnusedLocal
def _check_acr(data, *args):
    s = ('active_cylinder_ratios', 'engine_has_cylinder_deactivation')
    acr = data.get(s[0], (1,))

    has_acr = data.get(s[1], dfl.values.engine_has_cylinder_deactivation)

    if has_acr and len(acr) <= 1:
        msg = "Please since `engine_has_cylinder_deactivation` is True set " \
              "at least two `active_cylinder_ratios` or set False!"
        return s, msg
    elif not has_acr and len(acr) > 1:
        msg = "Please since there are %d `active_cylinder_ratios` set " \
              "`engine_has_cylinder_deactivation = True` " \
              "or remove the extra ratios!" % len(acr)
        return s, msg


# noinspection PyUnusedLocal
def _check_ki_factor(data, *args):
    s = 'has_periodically_regenerating_systems', 'ki_multiplicative', \
        'ki_additive'

    has_prs = data.get(s[0], dfl.values.has_periodically_regenerating_systems)

    if data.get(s[1], 1) > 1 and data.get(s[2], 0) > 0:
        msg = "Please since `ki_multiplicative` is > 1 and `ki_additive` " \
              "is > 0 set `ki_multiplicative = 1` or set `ki_additive = 0`!"
        return s[1:], msg
    elif not has_prs:
        if data.get(s[1], 1) > 1:
            msg = "Please since `ki_multiplicative` is > 1 set " \
                  "`has_periodically_regenerating_systems = True` or set " \
                  "`ki_multiplicative = 1`!"
            return s, msg
        elif data.get(s[2], 0) > 0:
            msg = "Please since `ki_additive` is > 0 set " \
                  "`has_periodically_regenerating_systems = True` or set " \
                  "`ki_additive = 0`!"
            return s, msg


# noinspection PyUnusedLocal
def _check_prediction_gears_not_mt(data, usage, stage, cycle, *args):
    s = ('gear_box_type', 'gears')
    gear_box_type = data.get(s[0], 'manual')
    if stage == 'prediction' and s[1] in data and gear_box_type != 'manual':
        msg = "`gears` cannot be provided when `gear_box_type` is '%s'." \
              " Hence, remove the `gears` or set `gear_box_type` to manual!"
        return s, msg % gear_box_type


@functools.lru_cache(None)
def _get_engine_model(outputs):
    from ...model.physical.engine import dsp
    return dsp.register(memo={}).shrink_dsp(outputs=outputs)


# noinspection PyUnusedLocal
def _check_lean_burn_tech(data, *args):
    s = ('has_lean_burn', 'ignition_type')
    it = _get_engine_model(s[1:])(data, outputs=s[1:]).get(s[1], None)
    has_lb = data.get(s[0], dfl.values.has_lean_burn)
    if has_lb and it not in ('positive', None):
        msg = "`has_lean_burn` cannot be enable with `ignition_type = '%s'`." \
              "Hence, set `has_lean_burn = False` or " \
              "set `ignition_type = 'positive'`!" % it
        return s, msg


# noinspection PyUnusedLocal
def _check_full_load(data, *args):
    s = ('idle_engine_speed_median', 'full_load_speeds')
    try:
        r = sh.selector(s, _get_engine_model(s)(data, s), output_type='list')
    except KeyError:
        return
    if r[0] < r[1][0]:
        msg = "You have not provided Full Load Curve values below %f RPMs. \n" \
              "This may cause issues in the simulation. Please start from " \
              "idle RPM (%f) or correct the " \
              "`idle_engine_speed_median = full_load_speeds[0]`!"
        return s, msg % (r[1][0], r[0])


# noinspection PyUnusedLocal
def _warn_vva(data, *args):
    s = ('engine_has_variable_valve_actuation', 'ignition_type')
    it = _get_engine_model(s[1:]).dispatch(data, outputs=s[1:]).get(s[1], None)
    has_vva = data.get(s[0], dfl.values.engine_has_variable_valve_actuation)
    if has_vva and it not in ('positive', None):
        msg = "Please, ensure that the input combination " \
              "`engine_has_variable_valve_actuation = True` and " \
              "`ignition_type = '%s'` is correct. If it is intentionally " \
              "added, you can neglect this warning. Otherwise, please " \
              "correct the input setting " \
              "`engine_has_variable_valve_actuation = False` or setting " \
              "`ignition_type = 'positive'`!" % it

        log.warning(msg)


# noinspection PyUnusedLocal
def _check_scr(data, *args):
    s = ('has_selective_catalytic_reduction', 'ignition_type')
    out = _get_engine_model(s[1:]).dispatch(data, outputs=s[1:])
    it = out.get(s[1], None)
    has_scr = data.get(s[0], dfl.values.has_selective_catalytic_reduction)
    if has_scr and it == 'positive':
        msg = "`has_selective_catalytic_reduction` cannot be enable with " \
              "`ignition_type = '%s'`." \
              "Hence, set `has_selective_catalytic_reduction = False` or " \
              "set `ignition_type = 'compression'`!" % it
        return s, msg


# noinspection PyUnusedLocal
def _check_has_torque_converter(data, *args):
    s = ('gear_box_type', 'has_torque_converter')
    if data.get(s[1], False) and data.get(s[0], '') == 'manual':
        msg = "`has_torque_converter` cannot be 'True' when " \
              "`gear_box_type` is 'manual'." \
              "Hence, set `has_torque_converter = False` or " \
              "set `gear_box_type != 'manual'`!"
        return s, msg


# noinspection PyUnusedLocal
def _check_relative_electric_energy_change(data, *args):
    c = ('relative_electric_energy_change', 'transition_cycle_index')
    try:
        reec, n = sh.selector(c, data, output_type='list')
        b = np.array(reec) < .04
        if not (b.shape[0] and b.sum() == 1 and b[-1]):
            msg = '{} must have only the last value <.04!'.format(c[0])
            return c[:-1], msg
        if len(reec) != n + 1:
            msg = '`len({}) != {} + 1`!'.format(*c)
            return c, msg
    except KeyError:  # `c` is not in `data`.
        pass


# noinspection PyUnusedLocal
def _check_gear_box(data, *args):
    c = ('gear_box_type', 'is_hybrid')
    try:
        gear_box_type, is_hybrid = sh.selector(c, data, output_type='list')
        if gear_box_type == 'planetary' and not is_hybrid:
            msg = "`gear_box_type` cannot be 'planetary' when " \
                  "`is_hybrid = False`." \
                  "Hence, set `gear_box_type != 'planetary'` or " \
                  "set `is_hybrid = True`!"
            return c, msg
    except KeyError:  # `c` is not in `data`.
        pass


def _hard_validation(data, usage, stage=None, cycle=None, *args):
    if usage in ('input', 'target', 'meta'):
        checks = (
            _check_sign_currents,
            _check_acr,
            _check_ki_factor,
            _check_prediction_gears_not_mt,
            _check_lean_burn_tech,
            _check_full_load,
            _warn_vva,
            _check_scr,
            _check_has_torque_converter,
            _check_relative_electric_energy_change,
            _check_gear_box
        )
        for check in checks:
            c = check(data, usage, stage, cycle, *args)
            if c:
                yield c


def hard_validation(inputs, errors):
    """
    Parse input data for the hard validation.

    :param inputs:
        Input data.
    :type inputs: dict

    :param errors:
        Errors container.
    :type errors: dict

    :return:
        Parsed input data and errors container.
    :rtype: dict, dict
    """
    from schema import SchemaError
    for k, v in sh.stack_nested_keys(inputs, depth=3):
        for c, msg in _hard_validation(v, *k):
            sh.get_nested_dicts(errors, *k)[c] = SchemaError([], [msg])
    return inputs, errors
