# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Core functions to predict the A/T gear shifting.
"""
import collections
from co2mpas.defaults import dfl


def define_gear_filter(
        change_gear_window_width=dfl.values.change_gear_window_width):
    """
    Defines a gear filter function.

    :param change_gear_window_width:
        Time window used to apply gear change filters [s].
    :type change_gear_window_width: float

    :return:
        Gear filter function.
    :rtype: callable
    """
    import numpy as np
    from co2mpas.utils import median_filter, clear_fluctuations

    def gear_filter(times, gears):
        """
        Filter the gears to remove oscillations.

        :param times:
            Time vector [s].
        :type times: numpy.array

        :param gears:
            Gear vector [-].
        :type gears: numpy.array

        :return:
            Filtered gears [-].
        :rtype: numpy.array
        """

        gears = median_filter(
            times, gears.astype(float), change_gear_window_width
        )
        gears = clear_fluctuations(times, gears, change_gear_window_width)

        return np.asarray(gears, dtype=int)

    return gear_filter


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
            if b.sum() > 2:
                a = (v[b] if isinstance(v, np.ndarray) else v for v in args)
                self[i] = model_class().fit(*a)
            b = ~b

        if len(self) == 2 and set(self['cold']) == set(self['hot']):
            return self

    # noinspection PyTypeChecker,PyCallByClass
    def predict(self, *args, **kwargs):
        from .cmv import CMV
        return CMV.predict(self, *args, **kwargs)

    def init_gear(self, gears, times, velocities, accelerations, motive_powers,
                  engine_coolant_temperatures=None,
                  correct_gear=lambda g, *args: g):
        from co2mpas.utils import List
        if gears is None:
            gears = List(dtype=int)

        gen = {k: v.init_gear(
            gears, times, velocities, accelerations, motive_powers,
            engine_coolant_temperatures, correct_gear=correct_gear
        ) for k, v in self.items()}

        def _next(i):
            if times[i] < self.time_cold_hot_transition:
                return gen['cold'](i)
            return gen['hot'](i)

        return _next

    def init_speed(self, *args, **kwargs):
        return self['hot'].init_speed(*args, **kwargs)
