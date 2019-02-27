# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains functions to predict the A/T gear shifting.
"""
import collections


def prediction_gears_gsm(
        correct_gear, gear_filter, gsm, times, velocities, accelerations,
        motive_powers, cycle_type=None, velocity_speed_ratios=None,
        engine_coolant_temperatures=None):
    """
    Predicts gears with a gear shifting model (cmv or gspv or dtgs or mgs) [-].

    :param correct_gear:
        A function to correct the gear predicted.
    :type correct_gear: callable

    :param gear_filter:
        Gear filter function.
    :type gear_filter: callable

    :param cycle_type:
        Cycle type (WLTP or NEDC).
    :type cycle_type: str

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param gsm:
        A gear shifting model (cmv or gspv or dtgs).
    :type gsm: GSPV | CMV | DTGS

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array, optional

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Predicted gears.
    :rtype: numpy.array
    """

    if velocity_speed_ratios is not None and cycle_type is not None:
        from . import _upgrade_gsm
        gsm = _upgrade_gsm(gsm, velocity_speed_ratios, cycle_type)

    # noinspection PyArgumentList
    gears = gsm.predict(
        times, velocities, accelerations, motive_powers,
        engine_coolant_temperatures,
        correct_gear=correct_gear, gear_filter=gear_filter
    )
    return gears


# noinspection PyMissingOrEmptyDocstring
class GSMColdHot(collections.OrderedDict):
    def __init__(self, *args, time_cold_hot_transition=0.0):
        super(GSMColdHot, self).__init__(*args)
        self.time_cold_hot_transition = time_cold_hot_transition

    def __repr__(self):
        name = self.__class__.__name__
        items = [(k, v) for k, v in self.items()]
        s = '{}({}, time_cold_hot_transition={})'.format(
            name, items, self.time_cold_hot_transition
        )
        return s.replace('inf', "float('inf')")

    def fit(self, model_class, times, *args):
        import numpy as np
        self.clear()

        b = times <= self.time_cold_hot_transition

        for i in ['cold', 'hot']:
            a = (v[b] if isinstance(v, np.ndarray) else v for v in args)
            self[i] = model_class().fit(*a)
            b = ~b
        return self

    # noinspection PyTypeChecker,PyCallByClass
    def predict(self, *args, **kwargs):
        from .cmv import CMV
        return CMV.predict(self, *args, **kwargs)

    def yield_gear(self, times, velocities, accelerations, motive_powers,
                   engine_coolant_temperatures=None,
                   correct_gear=lambda i, g, *args: g[i], index=0, gears=None):
        import numpy as np
        if gears is None:
            gears = np.zeros_like(times, int)

        n = index + np.searchsorted(
            times[index:], self.time_cold_hot_transition
        )

        flag, temp = engine_coolant_temperatures is not None, None
        for i, j, k in [(index, n, 'cold'), (n, times.shape[0], 'hot')]:
            if flag:
                temp = engine_coolant_temperatures[:j]
            yield from self[k].yield_gear(
                times[:j], velocities[:j], accelerations[:j], motive_powers[:j],
                temp, correct_gear=correct_gear, index=int(i), gears=gears
            )

    def yield_speed(self, *args, **kwargs):
        yield from self['hot'].yield_speed(*args, **kwargs)