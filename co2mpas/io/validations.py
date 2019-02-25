#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides the CO2MPAS validation formulas.
"""

import numpy as np
import copy
import schedula as sh
import co2mpas.utils as co2_utl
from . import constants
import functools
import logging

log = logging.getLogger(__name__)


def select_declaration_data(data, diff=None):
    res = {}
    for k, v in sh.stack_nested_keys(constants.con_vals.DECLARATION_DATA):
        if v and sh.are_in_nested_dicts(data, *k):
            v = sh.get_nested_dicts(data, *k)
            sh.get_nested_dicts(res, *k, default=co2_utl.ret_v(v))

    if diff is not None:
        diff.clear()
        diff.update(v[0] for v in sh.stack_nested_keys(data, depth=4))
        it = (v[0] for v in sh.stack_nested_keys(res, depth=4))
        diff.difference_update(it)
    return res


def overwrite_declaration_config_data(data):
    config = constants.con_vals.DECLARATION_SELECTOR_CONFIG
    res = sh.combine_nested_dicts(data, depth=3)
    key = ('config', 'selector', 'all')

    d = copy.deepcopy(sh.get_nested_dicts(res, *key))

    for k, v in sh.stack_nested_keys(config):
        sh.get_nested_dicts(d, *k, default=co2_utl.ret_v(v))

    sh.get_nested_dicts(res, *key[:-1])[key[-1]] = d

    return res


def hard_validation(data, usage, stage=None, cycle=None, *args):
    if usage in ('input', 'target', 'meta'):
        checks = (
            _check_sign_currents,
            # _check_initial_temperature,
            _check_acr,
            _check_ki_factor,
            _check_prediction_gears_not_mt,
            _check_lean_burn_tech,
            _check_full_load,
            _warn_vva,
            _check_scr,
            _check_has_torque_converter
        )
        for check in checks:
            c = check(data, usage, stage, cycle, *args)
            if c:
                yield c


def _check_sign_currents(data, *args):
    c = ('battery_currents', 'alternator_currents')
    try:
        a = sh.selector(c, data, output_type='list')
        s = check_sign_currents(*a)
        if not all(s):
            s = ' and '.join([k for k, v in zip(c, s) if not v])
            msg = "Probably '{}' have the wrong sign!".format(s)
            return c, msg
    except KeyError:  # `c` is not in `data`.
        pass


def _check_initial_temperature(data, *args):
    t = ('initial_temperature', 'engine_coolant_temperatures',
         'engine_speeds_out', 'idle_engine_speed_median')
    try:
        a = sh.selector(t, data, output_type='list')
        if not check_initial_temperature(*a):
            msg = "Initial engine temperature outside permissible limits " \
                  "according to GTR!"
            return t, msg
    except KeyError:  # `t` is not in `data`.
        pass


def check_sign_currents(battery_currents, alternator_currents):
    """
    Checks if battery currents and alternator currents have the right signs.

    :param battery_currents:
        Low voltage battery current vector [A].
    :type battery_currents: numpy.array

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :return:
        If battery and alternator currents have the right signs.
    :rtype: (bool, bool)
    """

    b_c, a_c = battery_currents, alternator_currents

    a = co2_utl.reject_outliers(a_c, med=np.mean)[0]
    a = a <= constants.con_vals.MAX_VALIDATE_POS_CURR
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


def check_initial_temperature(
        initial_temperature, engine_coolant_temperatures,
        engine_speeds_out, idle_engine_speed_median):
    """
    Checks if initial temperature is valid according NEDC and WLTP regulations.

    :param initial_temperature:
        Engine initial temperature [°C]
    :type initial_temperature: float

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param idle_engine_speed_median:
        Engine speed idle median [RPM].
    :type idle_engine_speed_median: float

    :return:
        True if data pass the checks.
    :rtype: bool
    """
    con_vals = constants.con_vals
    idle = idle_engine_speed_median - con_vals.DELTA_RPM2VALIDATE_TEMP
    b = engine_speeds_out > idle
    i = co2_utl.argmax(b) + 1
    t = np.mean(engine_coolant_temperatures[:i])
    dT = abs(initial_temperature - t)
    return dT <= con_vals.MAX_VALIDATE_DTEMP and t <= con_vals.MAX_INITIAL_TEMP


def _check_ki_factor(data, *args):
    s = 'has_periodically_regenerating_systems', 'ki_multiplicative', \
        'ki_additive'

    from ..model.physical.defaults import dfl
    has_prs = data.get(s[0], dfl.values.has_periodically_regenerating_systems)

    if (data.get(s[1], 1) > 1 and data.get(s[2], 0) > 0):
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


def _check_acr(data, *args):
    s = ('active_cylinder_ratios', 'engine_has_cylinder_deactivation')
    acr = data.get(s[0], (1,))

    from ..model.physical.defaults import dfl
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


def _check_prediction_gears_not_mt(data, usage, stage, cycle, *args):
    s = ('gear_box_type', 'gears')
    gear_box_type = data.get(s[0], 'manual')
    if stage == 'prediction' and s[1] in data and gear_box_type != 'manual':
        msg = "`gears` cannot be provided when `gear_box_type` is '%s'." \
              " Hence, remove the `gears` or set `gear_box_type` to manual!"
        return s, msg % gear_box_type


@functools.lru_cache(None)
def _get_engine_model(outputs):
    from ..model.physical.engine import engine
    return engine().shrink_dsp(outputs=outputs)


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


def _check_lean_burn_tech(data, usage, stage, cycle, *args):
    s = ('has_lean_burn', 'ignition_type')
    it = _get_engine_model(s[1:]).dispatch(data, outputs=s[1:]).get(s[1], None)
    from ..model.physical.defaults import dfl
    has_lb = data.get(s[0], dfl.values.has_lean_burn)
    if has_lb and it not in ('positive', None):
        msg = "`has_lean_burn` cannot be enable with `ignition_type = '%s'`." \
              "Hence, set `has_lean_burn = False` or " \
              "set `ignition_type = 'positive'`!" % it
        return s, msg


def _warn_vva(data, usage, stage, cycle, *args):
    s = ('engine_has_variable_valve_actuation', 'ignition_type')
    it = _get_engine_model(s[1:]).dispatch(data, outputs=s[1:]).get(s[1], None)
    from ..model.physical.defaults import dfl
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


def _check_scr(data, usage, stage, cycle, *args):
    s = ('has_selective_catalytic_reduction', 'ignition_type')
    out = _get_engine_model(s[1:]).dispatch(data, outputs=s[1:])
    it = out.get(s[1], None)
    from ..model.physical.defaults import dfl
    has_scr = data.get(s[0], dfl.values.has_selective_catalytic_reduction)
    if has_scr and it == 'positive':
        msg = "`has_selective_catalytic_reduction` cannot be enable with " \
              "`ignition_type = '%s'`." \
              "Hence, set `has_selective_catalytic_reduction = False` or " \
              "set `ignition_type = 'compression'`!" % it
        return s, msg


def _check_has_torque_converter(data, *args):
    s = ('gear_box_type', 'has_torque_converter')
    if data.get(s[1], False) and data.get(s[0], '') == 'manual':
        msg = "`has_torque_converter` cannot be 'True' when " \
              "`gear_box_type` is 'manual'." \
              "Hence, set `has_torque_converter = False` or " \
              "set `gear_box_type != 'manual'`!"
        return s, msg
