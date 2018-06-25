# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions that model the basic mechanics of the wheels.
"""

import math
import numpy as np
import schedula as sh
import co2mpas.utils as co2_utl
from .gear_box import mechanical as gb_mec
# noinspection PyCompatibility
import regex
import logging
import schema

log = logging.getLogger(__name__)


def calculate_wheel_power(velocities, accelerations, road_loads, vehicle_mass):
    """
    Calculates the wheel power [kW].

    :param velocities:
        Velocity [km/h].
    :type velocities: numpy.array | float

    :param accelerations:
        Acceleration [m/s2].
    :type accelerations: numpy.array | float

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :return:
        Power at wheels [kW].
    :rtype: numpy.array | float
    """

    f0, f1, f2 = road_loads

    quadratic_term = f0 + (f1 + f2 * velocities) * velocities

    vel = velocities / 3600

    return (quadratic_term + 1.03 * vehicle_mass * accelerations) * vel


# noinspection PyIncorrectDocstring,SpellCheckingInspection
def calculate_wheel_torques(wheel_powers, wheel_speeds, coef=30000 / math.pi):
    """
    Calculates torque at the wheels [N*m].

    :param wheel_powers:
        Power at the wheels [kW].
    :type wheel_powers: numpy.array | float

    :param wheel_speeds:
        Rotating speed of the wheel [RPM].
    :type wheel_speeds: numpy.array | float

    :return:
        Torque at the wheels [N*m].
    :rtype: numpy.array | float
    """
    return np.where(wheel_speeds, wheel_powers / wheel_speeds * coef, 0)


def calculate_wheel_powers(wheel_torques, wheel_speeds):
    """
    Calculates power at the wheels [kW].

    :param wheel_torques:
        Torque at the wheel [N*m].
    :type wheel_torques: numpy.array | float

    :param wheel_speeds:
        Rotating speed of the wheel [RPM].
    :type wheel_speeds: numpy.array | float

    :return:
        Power at the wheels [kW].
    :rtype: numpy.array | float
    """

    return wheel_torques * wheel_speeds * (math.pi / 30000.0)


def _compile_speed_function(r_dynamic):
    c = 30.0 / (3.6 * math.pi * r_dynamic)

    def _wheel_speed(velocities):
        return velocities * c

    return _wheel_speed


def calculate_wheel_speeds(velocities, r_dynamic):
    """
    Calculates rotating speed of the wheels [RPM].

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array | float

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Rotating speed of the wheel [RPM].
    :rtype: numpy.array | float
    """
    return _compile_speed_function(r_dynamic)(velocities)


def identify_r_dynamic_v1(
        velocities, gears, engine_speeds_out, gear_box_ratios,
        final_drive_ratios, stop_velocity):
    """
    Identifies the dynamic radius of the wheels [m].

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int | float]

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int | float]

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Dynamic radius of the wheels [m].
    :rtype: float
    """

    svr = gb_mec.calculate_speed_velocity_ratios(
        gear_box_ratios, final_drive_ratios, 1.0)

    vsr = gb_mec.calculate_velocity_speed_ratios(svr)

    speed_x_r_dyn_ratios = gb_mec.calculate_gear_box_speeds_in(
        gears, velocities, vsr, stop_velocity
    )

    r_dynamic = speed_x_r_dyn_ratios / engine_speeds_out
    r_dynamic = r_dynamic[~np.isnan(r_dynamic)]
    r_dynamic = co2_utl.reject_outliers(r_dynamic)[0]

    return r_dynamic


def identify_r_dynamic_v2(
        times, velocities, accelerations, r_wheels, engine_speeds_out,
        gear_box_ratios, final_drive_ratios, idle_engine_speed, stop_velocity,
        plateau_acceleration, change_gear_window_width):
    """
    Identifies the dynamic radius of the wheels [m].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param accelerations:
        Vehicle acceleration [m/s2].
    :type accelerations: numpy.array

    :param r_wheels:
        Radius of the wheels [m].
    :type r_wheels: float

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int | float]

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int | float]

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :param plateau_acceleration:
        Maximum acceleration to be at constant velocity [m/s2].
    :type plateau_acceleration: float

    :param change_gear_window_width:
        Time window used to apply gear change filters [s].
    :type change_gear_window_width: float

    :return:
        Dynamic radius of the wheels [m].
    :rtype: float
    """

    svr = gb_mec.calculate_speed_velocity_ratios(
        gear_box_ratios, final_drive_ratios, r_wheels
    )

    gears = gb_mec.identify_gears(
        times, velocities, accelerations, engine_speeds_out,
        gb_mec.calculate_velocity_speed_ratios(svr), stop_velocity,
        plateau_acceleration, change_gear_window_width, idle_engine_speed
    )

    r_dynamic = identify_r_dynamic_v1(
        velocities, gears, engine_speeds_out, gear_box_ratios,
        final_drive_ratios, stop_velocity
    )

    return r_dynamic


def identify_r_dynamic(
        velocity_speed_ratios, gear_box_ratios, final_drive_ratios):
    """
    Identifies the dynamic radius of the wheels [m].

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict[int | float]

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int | float]

    :return:
        Dynamic radius of the wheels [m].
    :rtype: float
    """

    svr = gb_mec.calculate_speed_velocity_ratios(
        gear_box_ratios, final_drive_ratios, 1
    )

    r = [svr[k] * vs for k, vs in velocity_speed_ratios.items() if k]

    r_dynamic = co2_utl.reject_outliers(r)[0]

    return r_dynamic


def identify_tyre_dynamic_rolling_coefficient(r_wheels, r_dynamic):
    """
    Identifies the dynamic rolling coefficient [-].

    :param r_wheels:
        Radius of the wheels [m].
    :type r_wheels: float

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Dynamic rolling coefficient [-].
    :rtype: float
    """

    return r_dynamic / r_wheels


def calculate_r_dynamic(r_wheels, tyre_dynamic_rolling_coefficient):
    """
    Calculates the dynamic radius of the wheels [m].

    :param r_wheels:
        Radius of the wheels [m].
    :type r_wheels: float

    :param tyre_dynamic_rolling_coefficient:
        Dynamic rolling coefficient [-].
    :type tyre_dynamic_rolling_coefficient: float

    :return:
        Dynamic radius of the wheels [m].
    :rtype: float
    """

    return tyre_dynamic_rolling_coefficient * r_wheels


_re_tyre_code_iso = regex.compile(
    r"""
    ^(?P<use>([a-z]){1,2})?\s*
    (?P<nominal_section_width>(\d){3})\s*
    \/\s*
    (?P<aspect_ratio>(\d){2,3})?
    ((\s*(?P<carcass>[a-z])\s*)|\s+)
    (?P<rim_diameter>(\d){1,2}(\.(\d){1,2})?)
    (\s+(?P<use>C))?
    (\s+(?P<load_index>(\d){2,3}(/(\d){2,3})?)\s*
     (?P<speed_rating>(\([a-z]\)|[a-z]\d?)))?\s*
    (\s*((?P<load_range>[a-z])(^| )))?
    (\s+(?P<additional_marks>.*))?$
    """, regex.IGNORECASE | regex.X | regex.DOTALL)

_re_tyre_code_numeric = regex.compile(
    r"""
    ^((?P<diameter>(\d){2})\s*x\s*)?
    (?P<nominal_section_width>(\d){1,2}(\.(\d){1,2})?)\s*
    ((\s*(?P<carcass>([a-z]|-))\s*)|\s+)
    (?P<rim_diameter>(\d){2}(\.(\d){1,2})?)\s*
    (?P<use>(LT|C))\s*
    ((?P<load_index>(\d){2,3}(/(\d){2,3})?)\s*
     (?P<speed_rating>(\([a-z]\)|[a-z]\d?)))?\s*
    (\s*((?P<load_range>[a-z])(^| )))?
    (\s+(?P<additional_marks>.*))?$
    """, regex.IGNORECASE | regex.X | regex.DOTALL)


# noinspection PyUnusedLocal
def _format_tyre_code(
        nominal_section_width, rim_diameter, aspect_ratio=0, use='', carcass='',
        load_index='', speed_rating='', additional_marks='', load_range='',
        code='iso', diameter=None, **kw):
    if code == 'iso':
        parts = (
            '%s%d/%d%s%d' % (use, nominal_section_width, aspect_ratio,
                             carcass or ' ', rim_diameter),
        )
    else:
        diameter = '%.2fx' % diameter if diameter is not None else ''
        parts = (
            '%s%.2f%s%.2f %s' % (diameter, nominal_section_width,
                                 carcass or ' ', rim_diameter, use),
        )

    parts += (
        '%s%s' % (load_index, speed_rating),
        load_range,
        additional_marks
    )
    return ' '.join(p for p in parts if p)


def _format_tyre_dimensions(tyre_dimensions):
    frt = schema.Schema({
        schema.Optional('additional_marks'): schema.Use(str),
        'aspect_ratio': schema.Use(float),
        schema.Optional('carcass'): schema.Use(str),
        'rim_diameter': schema.Use(float),
        schema.Optional('diameter'): schema.Use(float),
        schema.Optional('load_index'): schema.Use(str),
        schema.Optional('load_range'): schema.Use(str),
        'nominal_section_width': schema.Use(float),
        schema.Optional('speed_rating'): schema.Use(str),
        schema.Optional('use'): schema.Use(str),
        schema.Optional('code'): schema.Use(str),
    })
    m = {k: v for k, v in tyre_dimensions.items() if v is not None}
    return frt.validate(m)


def define_tyre_code(tyre_dimensions):
    """
    Returns the tyre code from the tyre dimensions.

    :param tyre_dimensions:
        Tyre dimensions.

        .. note:: The fields are : use, nominal_section_width, aspect_ratio,
           carcass, diameter, load_index, speed_rating, and additional_marks.
    :type tyre_dimensions: dict

    :return:
        Tyre code (e.g.,P225/70R14).
    :rtype: str
    """
    return _format_tyre_code(**tyre_dimensions)


def calculate_r_wheels(tyre_dimensions):
    """
    Calculates the radius of the wheels [m] from the tyre dimensions.

    :param tyre_dimensions:
        Tyre dimensions.

        .. note:: The fields are : use, nominal_section_width, aspect_ratio,
           carcass, diameter, load_index, speed_rating, and additional_marks.
    :type tyre_dimensions: dict

    :return:
        Radius of the wheels [m].
    :rtype: float
    """
    if 'diameter' in tyre_dimensions:
        return tyre_dimensions['diameter'] * 0.0254  # Diameter is in inches.
    a = tyre_dimensions['aspect_ratio'] / 100  # Aspect ratio is Height/Width.
    w = tyre_dimensions['nominal_section_width']
    if tyre_dimensions.get('code', 'iso') == 'iso':
        w /= 1000  # Width is in mm.
    else:
        w *= 0.0254  # Width is in inches.

    dr = tyre_dimensions['rim_diameter'] * 0.0254  # Rim is in inches.
    return a * w + dr / 2


def default_tyre_code(r_dynamic):
    """
    Return one of the most popular tyre code according to the r dynamic.

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Tyre code (e.g.,P225/70R14).
    :rtype: str
    """

    pop = (
        '165/65R13', '155/70R13', '165/70R13', '165/60R14', '185/60R14',
        '155/65R14', '165/65R14', '175/65R14', '185/65R14', '165/70R14',
        '175/70R14', '195/50R15', '185/55R15', '195/55R15', '185/60R15',
        '195/60R15', '205/60R15', '175/65R15', '185/65R15', '195/65R15',
        '195/70R15', '195/45R16', '205/45R16', '205/50R16', '195/55R16',
        '205/55R16', '215/55R16', '205/60R16', '215/60R16', '215/65R16',
        '205/40R17', '245/40R17', '205/45R17', '215/45R17', '225/45R17',
        '235/45R17', '205/50R17', '215/50R17', '225/50R17', '215/55R17',
        '225/55R17', '215/60R17', '225/65R17', '235/65R17', '225/40R18',
        '235/40R18', '245/40R18', '225/45R18', '235/60R18', '255/35R19'
    )

    def _key_func(c):
        r = calculate_r_wheels(calculate_tyre_dimensions(c))
        return r <= r_dynamic, (r - r_dynamic) ** 2

    return min(pop, key=_key_func)


def calculate_tyre_dimensions(tyre_code):
    """
    Calculates the tyre dimensions from the tyre code.

    :param tyre_code:
        Tyre code (e.g.,P225/70R14).
    :type tyre_code: str

    :return:
        Tyre dimensions.
    :rtype: dict
    """
    it = ('iso', _re_tyre_code_iso), ('numeric', _re_tyre_code_numeric)
    for c, _r in it:
        try:
            m = _r.match(tyre_code).groupdict()
            m['code'] = c
            if c == 'numeric' and 'aspect_ratio' not in m:
                b = m['nominal_section_width'].split('.')[-1][-1] == '5'
                m['aspect_ratio'] = '82' if b else '92'
            return _format_tyre_dimensions(m)
        except (AttributeError, schema.SchemaError):
            pass
    raise ValueError('Invalid tyre code: %s', tyre_code)


# noinspection PyMissingOrEmptyDocstring
class WheelsModel:
    key_outputs = [
        'wheel_speeds',
        'wheel_powers',
        'wheel_torques'
    ]

    types = {float: set(key_outputs)}

    def __init__(self, r_dynamic=None, outputs=None):
        self.r_dynamic = r_dynamic
        self._outputs = outputs
        self.outputs = None

    def __call__(self, times, *args, **kwargs):
        self.set_outputs(times.shape[0])
        for _ in self.yield_results(times, *args, **kwargs):
            pass
        return sh.selector(self.key_outputs, self.outputs, output_type='list')

    def yield_speed(self, velocities):
        key = 'wheel_speeds'
        if self._outputs is not None and key in self._outputs:
            yield from self._outputs[key]
        else:
            yield from map(_compile_speed_function(self.r_dynamic), velocities)

    def yield_power(self, motive_powers):
        key = 'wheel_powers'
        if self._outputs is not None and key in self._outputs:
            yield from self._outputs[key]
        else:
            yield from motive_powers

    def yield_torque(self, wheel_powers, wheel_speeds):
        key = 'wheel_torques'
        if self._outputs is not None and key in self._outputs:
            yield from self._outputs[key]
        else:
            for v in zip(wheel_powers, wheel_speeds):
                yield calculate_wheel_torques(*v)

    def set_outputs(self, n, outputs=None):
        if outputs is None:
            outputs = {}
        outputs.update(self._outputs or {})

        for t, names in self.types.items():
            names = names - set(outputs)
            if names:
                outputs.update(zip(names, np.empty((len(names), n), dtype=t)))

        self.outputs = outputs

    def yield_results(self, velocities, motive_powers):
        outputs = self.outputs

        s_gen = self.yield_speed(velocities)

        p_gen = self.yield_power(motive_powers)

        t_gen = self.yield_torque(
            outputs['wheel_powers'], outputs['wheel_speeds']
        )

        for i, (s, p) in enumerate(zip(s_gen, p_gen)):
            outputs['wheel_speeds'][i] = s
            outputs['wheel_powers'][i] = p
            outputs['wheel_torques'][i] = t = next(t_gen)
            yield s, p, t


def define_fake_wheels_prediction_model(
        wheel_speeds, wheel_powers, wheel_torques):
    """
    Defines a fake wheels prediction model.

    :param wheel_speeds:
        Rotating speed of the wheel [RPM].
    :type wheel_speeds: numpy.array

    :param wheel_powers:
        Power at the wheels [kW].
    :type wheel_powers: numpy.array

    :param wheel_torques:
        Torque at the wheel [N*m].
    :type wheel_torques: numpy.array

    :return:
        Wheels prediction model.
    :rtype: WheelsModel
    """
    model = WheelsModel(outputs={
        'wheel_speeds': wheel_speeds,
        'wheel_powers': wheel_powers,
        'wheel_torques': wheel_torques
    })
    return model


def define_wheels_prediction_model(r_dynamic):
    """
    Defines the wheels prediction model.

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Wheels prediction model.
    :rtype: WheelsModel
    """
    return WheelsModel(r_dynamic)


def wheels():
    """
    Defines the wheels model.

    .. dispatcher:: d

        >>> d = wheels()

    :return:
        The wheels model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Wheel model',
        description='It models the wheel dynamics.'
    )

    d.add_function(
        function=calculate_wheel_torques,
        inputs=['wheel_powers', 'wheel_speeds'],
        outputs=['wheel_torques']
    )

    d.add_function(
        function=calculate_wheel_powers,
        inputs=['wheel_torques', 'wheel_speeds'],
        outputs=['wheel_powers']
    )

    d.add_function(
        function=calculate_wheel_speeds,
        inputs=['velocities', 'r_dynamic'],
        outputs=['wheel_speeds']
    )

    d.add_function(
        function=identify_r_dynamic,
        inputs=['velocity_speed_ratios', 'gear_box_ratios',
                'final_drive_ratios'],
        outputs=['r_dynamic']
    )

    d.add_function(
        function=identify_r_dynamic_v1,
        inputs=['velocities', 'gears', 'engine_speeds_out', 'gear_box_ratios',
                'final_drive_ratios', 'stop_velocity'],
        outputs=['r_dynamic'],
        weight=10
    )

    from .defaults import dfl
    d.add_data(
        data_id='stop_velocity',
        default_value=dfl.values.stop_velocity
    )

    d.add_data(
        data_id='plateau_acceleration',
        default_value=dfl.values.plateau_acceleration
    )

    d.add_data(
        data_id='change_gear_window_width',
        default_value=dfl.values.change_gear_window_width
    )

    d.add_function(
        function=calculate_tyre_dimensions,
        inputs=['tyre_code'],
        outputs=['tyre_dimensions']
    )

    d.add_function(
        function=calculate_r_wheels,
        inputs=['tyre_dimensions'],
        outputs=['r_wheels']
    )

    d.add_function(
        function=define_tyre_code,
        inputs=['tyre_dimensions'],
        outputs=['tyre_code']
    )

    d.add_function(
        function=default_tyre_code,
        inputs=['r_dynamic'],
        outputs=['tyre_code'],
        weight=5
    )

    d.add_data(
        data_id='tyre_dynamic_rolling_coefficient',
        default_value=dfl.values.tyre_dynamic_rolling_coefficient,
        initial_dist=50
    )

    d.add_function(
        function=calculate_r_dynamic,
        inputs=['r_wheels', 'tyre_dynamic_rolling_coefficient'],
        outputs=['r_dynamic']
    )

    d.add_function(
        function=identify_tyre_dynamic_rolling_coefficient,
        inputs=['r_wheels', 'r_dynamic'],
        outputs=['tyre_dynamic_rolling_coefficient']
    )

    d.add_function(
        function=identify_r_dynamic_v2,
        inputs=['times', 'velocities', 'accelerations', 'r_wheels',
                'engine_speeds_out', 'gear_box_ratios', 'final_drive_ratios',
                'idle_engine_speed', 'stop_velocity', 'plateau_acceleration',
                'change_gear_window_width'],
        outputs=['r_dynamic'],
        weight=11
    )

    d.add_function(
        function=sh.bypass,
        inputs=['motive_powers'],
        outputs=['wheel_powers']
    )

    d.add_function(
        function=define_fake_wheels_prediction_model,
        inputs=['wheel_speeds', 'wheel_powers', 'wheel_torques'],
        outputs=['wheels_prediction_model']
    )

    d.add_function(
        function=define_wheels_prediction_model,
        inputs=['r_dynamic'],
        outputs=['wheels_prediction_model'],
        weight=4000
    )

    return d
