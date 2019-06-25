# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the final drive.
"""

import logging
import collections
import numpy as np
import schedula as sh
from .defaults import dfl
from co2mpas.utils import BaseModel

log = logging.getLogger(__name__)

dsp = sh.BlueDispatcher(
    name='Final drive', description='Models the final drive.'
)
dsp.add_data('final_drive_ratio', dfl.values.final_drive_ratio)


@sh.add_function(dsp, inputs_kwargs=True, outputs=['final_drive_ratios'])
def calculate_final_drive_ratios(final_drive_ratio, n_gears=1):
    """
    Defines final drive ratios for each gear [-].

    :param final_drive_ratio:
        Final drive ratio [-].
    :type final_drive_ratio: float

    :param n_gears:
        Number of gears [-].
    :type n_gears: int, optional

    :return:
        Final drive ratios [-].
    :rtype: dict
    """
    d = collections.defaultdict(lambda: final_drive_ratio)
    d.update(dict.fromkeys(range(0, int(n_gears + 1)), final_drive_ratio))
    return d


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def is_cvt(gear_box_type, *args):
    return gear_box_type == 'cvt'


dsp.add_function(
    function=sh.add_args(calculate_final_drive_ratios),
    inputs=['gear_box_type', 'final_drive_ratio'],
    outputs=['final_drive_ratios'],
    input_domain=is_cvt
)


@sh.add_function(dsp, outputs=['final_drive_ratio_vector'])
def calculate_final_drive_ratio_vector(final_drive_ratios, gears):
    """
    Calculates the final drive ratio vector [-].

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :return:
        Final drive ratio vector [-].
    :rtype: numpy.array
    """
    d = collections.defaultdict(lambda: dfl.values.final_drive_ratio)
    d.update(final_drive_ratios)
    return np.vectorize(lambda k: d[k])(gears)


@sh.add_function(dsp, outputs=['final_drive_speeds_in'])
def calculate_final_drive_speeds_in(
        final_drive_speeds_out, final_drive_ratio_vector):
    """
    Calculates final drive speed [RPM].

    :param final_drive_speeds_out:
        Rotating speed of the wheel [RPM].
    :type final_drive_speeds_out: numpy.array | float

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array | float

    :return:
        Final drive speed in [RPM].
    :rtype: numpy.array | float
    """
    return final_drive_speeds_out * final_drive_ratio_vector


dsp.add_data('n_wheel_drive', dfl.values.n_wheel_drive)


@sh.add_function(dsp, outputs=['final_drive_efficiency'])
def default_final_drive_efficiency(n_wheel_drive):
    from asteval import Interpreter as Interp
    formula = dfl.functions.default_final_drive_efficiency.formula
    return Interp().eval(formula)(n_wheel_drive)


@sh.add_function(dsp, outputs=['final_drive_powers_in'])
def calculate_final_drive_powers_in(
        final_drive_powers_out, final_drive_efficiency):
    eff, p = final_drive_efficiency, final_drive_powers_out
    return np.where(p > 0, eff, 1 / eff) * p


@sh.add_function(dsp, outputs=['final_drive_torques_in'])
def calculate_final_drive_torques_in(
        final_drive_powers_in, final_drive_speeds_in):
    from .wheels import calculate_wheel_torques as func
    return func(final_drive_powers_in, final_drive_speeds_in)


# noinspection PyUnusedLocal
def domain_final_drive_torque_losses_v1(n_dyno_axes, n_wheel_drive, *args):
    """
    Check the validity of number of wheel drive respect to the dyno axes
    assuming 2 wheels per axes.

    :param n_dyno_axes:
        Number of dyno axes [-].
    :type n_dyno_axes: int

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :return:
        True and log a waring if `n_wheel_drive` does not respect the domain.
    :rtype: bool
    """

    if n_dyno_axes < n_wheel_drive / 2:
        msg = 'WARNING: n_dyno_axes(%d) < n_wheel_drive(%d) / 2!'
        log.warning(msg, n_dyno_axes, n_wheel_drive)
    return True


# noinspection PyMissingOrEmptyDocstring
class FinalDriveModel(BaseModel):
    key_outputs = (
        'final_drive_ratio_vector',
        'final_drive_speeds_in',
        'final_drive_torque_losses',
        'final_drive_torques_in',
        'final_drive_efficiencies',
        'final_drive_powers_in'
    )

    types = {float: set(key_outputs)}

    def __init__(self, final_drive_ratios=None, final_drive_torque_loss=None,
                 n_wheel_drive=None, final_drive_efficiency=None, outputs=None):
        self.final_drive_ratios = collections.defaultdict(
            lambda: dfl.values.final_drive_ratio
        )
        self.final_drive_ratios.update(final_drive_ratios or {})
        self.final_drive_torque_loss = final_drive_torque_loss
        self.n_wheel_drive = n_wheel_drive
        self.final_drive_efficiency = final_drive_efficiency
        super(FinalDriveModel, self).__init__(outputs)

    def init_ratio(self, gears):
        key = 'final_drive_ratio_vector'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]
        d = collections.defaultdict(lambda: dfl.values.final_drive_ratio)
        d.update(self.final_drive_ratios)
        return lambda i: d[gears[i - 1] if i else 0]

    def init_speed(self, final_drive_speeds_out, final_drive_ratio_vector):
        key = 'final_drive_speeds_in'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]
        speeds, ratios = final_drive_speeds_out, final_drive_ratio_vector

        def _next(i):
            return calculate_final_drive_speeds_in(speeds[i], ratios[i])

        return _next

    def init_torque(self, final_drive_torques_out, final_drive_ratio_vector):
        keys = ['final_drive_torque_losses', 'final_drive_torques_in']

        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            losses, tor = sh.selector(keys, self._outputs, output_type='list')
            return lambda i: (losses[i], tor[i])

        if self.final_drive_torque_loss:
            # noinspection PyUnusedLocal
            def t_loss(*args):
                return self.final_drive_torque_loss
        else:
            t_loss = _compile_torque_losses_function(
                self.n_wheel_drive, self.final_drive_efficiency
            )

        ratios, torques = final_drive_ratio_vector, final_drive_torques_out

        def _next(i):
            r, t = ratios[i], torques[i]
            loss = t_loss(r, t)
            return loss, calculate_final_drive_torques_in(t, r, loss)

        return _next

    def init_power(self, final_drive_torques_out, final_drive_ratio_vector,
                   final_drive_torques_in, final_drive_powers_out):
        keys = ['final_drive_efficiencies', 'final_drive_powers_in']

        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            eff, pwr = sh.selector(keys, self._outputs, output_type='list')
            return lambda i: (eff[i], pwr[i])

        t_out, ratios, t_in, p_out = (
            final_drive_torques_out, final_drive_ratio_vector,
            final_drive_torques_in, final_drive_powers_out
        )

        def _next(i):
            to, r, ti, po = t_out[i], ratios[i], t_in[i], p_out[i]
            e = calculate_final_drive_efficiencies(to, r, ti)
            return e, calculate_final_drive_powers_in(po, e)

        return _next

    def init_results(self, gears, final_drive_speeds_out,
                     final_drive_torques_out, final_drive_powers_out):
        out = self.outputs
        ratio, speeds, losses, torques, eff, powers = (
            out['final_drive_ratio_vector'], out['final_drive_speeds_in'],
            out['final_drive_torque_losses'], out['final_drive_torques_in'],
            out['final_drive_efficiencies'], out['final_drive_powers_in']

        )
        r_gen = self.init_ratio(gears)
        s_gen = self.init_speed(final_drive_speeds_out, ratio)
        t_gen = self.init_torque(final_drive_torques_out, ratio)
        p_gen = self.init_power(
            final_drive_torques_out, ratio, torques, final_drive_powers_out
        )

        def _next(i):
            ratio[i] = r = r_gen(i)
            speeds[i] = s = s_gen(i)
            losses[i], torques[i] = l, t = t_gen(i)
            eff[i], powers[i] = e, p = p_gen(i)
            return r, s, l, t, e, p

        return _next


@sh.add_function(dsp, outputs=['final_drive_prediction_model'])
def define_fake_final_drive_prediction_model(
        final_drive_ratio_vector, final_drive_speeds_in,
        final_drive_torque_losses, final_drive_torques_in,
        final_drive_efficiencies, final_drive_powers_in):
    """
    Defines a fake final drive prediction model.

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array

    :param final_drive_speeds_in:
        Final drive speed in [RPM].
    :type final_drive_speeds_in: numpy.array

    :param final_drive_torque_losses:
        Final drive torque losses [N*m].
    :type final_drive_torque_losses: numpy.array

    :param final_drive_torques_in:
        Final drive torque in [N*m].
    :type final_drive_torques_in: numpy.array

    :param final_drive_efficiencies:
        Final drive torque efficiency vector [-].
    :type final_drive_efficiencies: numpy.array

    :param final_drive_powers_in:
        Final drive power in [kW].
    :type final_drive_powers_in: numpy.array

    :return:
        Final drive prediction model.
    :rtype: FinalDriveModel
    """
    model = FinalDriveModel(outputs={
        'final_drive_ratio_vector': final_drive_ratio_vector,
        'final_drive_speeds_in': final_drive_speeds_in,
        'final_drive_torque_losses': final_drive_torque_losses,
        'final_drive_torques_in': final_drive_torques_in,
        'final_drive_efficiencies': final_drive_efficiencies,
        'final_drive_powers_in': final_drive_powers_in,
    })

    return model


@sh.add_function(dsp, outputs=['final_drive_prediction_model'], weight=4000)
def define_final_drive_prediction_model(
        final_drive_ratios, final_drive_torque_loss, n_wheel_drive,
        final_drive_efficiency):
    """
    Defines the final drive prediction model.

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int, float | int]

    :param final_drive_torque_loss:
        Constant Final drive torque loss [N*m].
    :type final_drive_torque_loss: float

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :param final_drive_efficiency:
        Final drive efficiency [-].
    :type final_drive_efficiency: float

    :return:
        Final drive prediction model.
    :rtype: FinalDriveModel
    """
    model = FinalDriveModel(
        final_drive_ratios, final_drive_torque_loss, n_wheel_drive,
        final_drive_efficiency
    )
    return model
