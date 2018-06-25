# -*- coding: utf-8 -*-
#
# Copyright 2015-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It contains functions that model the basic mechanics of the final drive.
"""

import schedula as sh
import logging
import collections
import numpy as np

log = logging.getLogger(__name__)


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


def calculate_final_drive_torque_losses(
        final_drive_torques_out, final_drive_torque_loss):
    """
    Calculates final drive torque losses [N*m].

    :param final_drive_torques_out:
        Torque at the wheels [N*m].
    :type final_drive_torques_out: numpy.array

    :param final_drive_torque_loss:
        Constant Final drive torque loss [N*m].
    :type final_drive_torque_loss: float

    :return:
        Final drive torque losses [N*m].
    :rtype: numpy.array
    """

    return np.tile((final_drive_torque_loss,), final_drive_torques_out.shape)


def _compile_torque_losses_function(n_wheel_drive, final_drive_efficiency):
    d = final_drive_efficiency - (n_wheel_drive - 2) / 100
    n = (1 - d)

    # noinspection PyMissingOrEmptyDocstring
    def losses(final_drive_ratio_vector, final_drive_torques_out):
        return n / (d * final_drive_ratio_vector) * final_drive_torques_out

    return losses


def calculate_final_drive_torque_losses_v1(
        n_wheel_drive, final_drive_torques_out, final_drive_ratio_vector,
        final_drive_efficiency):
    """
    Calculates final drive torque losses [N*m].

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :param final_drive_torques_out:
        Torque at the wheels [N*m].
    :type final_drive_torques_out: numpy.array | float

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array | float

    :param final_drive_efficiency:
        Final drive efficiency [-].
    :type final_drive_efficiency: float

    :return:
        Final drive torque losses [N*m].
    :rtype: numpy.array | float
    """
    fun = _compile_torque_losses_function(n_wheel_drive, final_drive_efficiency)
    return fun(final_drive_ratio_vector, final_drive_torques_out)


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


def calculate_final_drive_torques_in(
        final_drive_torques_out, final_drive_ratio_vector,
        final_drive_torque_losses):
    """
    Calculates final drive torque [N*m].

    :param final_drive_torques_out:
        Torque at the wheels [N*m].
    :type final_drive_torques_out: numpy.array | float

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array | float

    :param final_drive_torque_losses:
        Final drive torque losses [N*m].
    :type final_drive_torque_losses: numpy.array | float

    :return:
        Final drive torque in [N*m].
    :rtype: numpy.array | float
    """

    t = final_drive_torques_out / final_drive_ratio_vector

    return t + final_drive_torque_losses


def calculate_final_drive_efficiencies(
        final_drive_torques_out, final_drive_ratio_vector,
        final_drive_torques_in):
    """
    Calculates final drive efficiency [-].

    :param final_drive_torques_out:
        Torque at the wheels [N*m].
    :type final_drive_torques_out: numpy.array | float

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array | float

    :param final_drive_torques_in:
        Final drive torque in [N*m].
    :type final_drive_torques_in: numpy.array | float

    :return:
        Final drive torque efficiency vector [-].
    :rtype: numpy.array
    """

    ratio = final_drive_ratio_vector
    eff = np.where(
        (final_drive_torques_out == 0) & (final_drive_torques_in == 0),
        1, final_drive_torques_out / (ratio * final_drive_torques_in)
    )

    return np.nan_to_num(eff)


def calculate_final_drive_powers_in(
        final_drive_powers_out, final_drive_efficiencies):
    """
    Calculates final drive power [kW].

    :param final_drive_powers_out:
        Power at the wheels [kW].
    :type final_drive_powers_out: numpy.array | float

    :param final_drive_efficiencies:
        Final drive torque efficiency vector [-].
    :type final_drive_efficiencies: numpy.array | float

    :return:
        Final drive power in [kW].
    :rtype: numpy.array | float
    """

    return final_drive_powers_out / final_drive_efficiencies


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


def calculate_final_drive_ratio_vector(final_drive_ratios, gears):
    """
    Calculates the final drive ratio vector [-].

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int | float]

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :return:
        Final drive ratio vector [-].
    :rtype: numpy.array
    """
    from .defaults import dfl
    d = collections.defaultdict(lambda: dfl.values.final_drive_ratio)
    d.update(final_drive_ratios)
    return np.vectorize(lambda k: d[k])(np.append([0], gears[:-1]))


# noinspection PyUnusedLocal,PyMissingOrEmptyDocstring
def is_cvt(gear_box_type, *args):
    return gear_box_type == 'cvt'


# noinspection PyMissingOrEmptyDocstring
class FinalDriveModel:
    key_outputs = [
        'final_drive_ratio_vector',
        'final_drive_speeds_in',
        'final_drive_torque_losses',
        'final_drive_torques_in',
        'final_drive_efficiencies',
        'final_drive_powers_in'
    ]

    types = {float: set(key_outputs)}

    def __init__(self, final_drive_ratios=None, final_drive_torque_loss=None,
                 n_wheel_drive=None, final_drive_efficiency=None, outputs=None):
        from .defaults import dfl
        self.final_drive_ratios = collections.defaultdict(
            lambda: dfl.values.final_drive_ratio
        )
        self.final_drive_ratios.update(final_drive_ratios or {})
        self.final_drive_torque_loss = final_drive_torque_loss
        self.n_wheel_drive = n_wheel_drive
        self.final_drive_efficiency = final_drive_efficiency
        self._outputs = outputs
        self.outputs = None

    def __call__(self, times, *args, **kwargs):
        self.set_outputs(times.shape[0])
        for _ in self.yield_results(times, *args, **kwargs):
            pass
        return sh.selector(self.key_outputs, self.outputs, output_type='list')

    def yield_ratio(self, gears):
        key = 'final_drive_ratio_vector'
        if self._outputs is not None and key in self._outputs:
            yield from self._outputs[key]
        else:
            get = self.final_drive_ratios.get
            yield get(0)
            for g in gears[:-1]:
                yield get(g)

    def yield_speed(self, final_drive_speeds_out, final_drive_ratio_vector):
        key = 'final_drive_speeds_in'
        if self._outputs is not None and key in self._outputs:
            yield from self._outputs[key]
        else:
            for v in zip(final_drive_speeds_out, final_drive_ratio_vector):
                yield calculate_final_drive_speeds_in(*v)

    def yield_torque(self, final_drive_torques_out, final_drive_ratio_vector):
        keys = ['final_drive_torque_losses', 'final_drive_torques_in']

        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            yield from zip(*sh.selector(
                keys, self._outputs, output_type='list'
            ))
        else:
            if self.final_drive_torque_loss:
                # noinspection PyUnusedLocal
                def t_loss(*args):
                    return self.final_drive_torque_loss
            else:
                t_loss = _compile_torque_losses_function(
                    self.n_wheel_drive, self.final_drive_efficiency
                )
            for r, t in zip(final_drive_ratio_vector, final_drive_torques_out):
                loss = t_loss(r, t)
                yield loss, calculate_final_drive_torques_in(t, r, loss)

    def yield_power(self, final_drive_torques_out, final_drive_ratio_vector,
                    final_drive_torques_in, final_drive_powers_out):
        keys = ['final_drive_efficiencies', 'final_drive_powers_in']

        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            yield from zip(*sh.selector(
                keys, self._outputs, output_type='list'
            ))
        else:
            it = zip(
                final_drive_torques_out, final_drive_ratio_vector,
                final_drive_torques_in, final_drive_powers_out
            )
            for to, r, ti, po in it:
                eff = calculate_final_drive_efficiencies(to, r, ti)
                yield eff, calculate_final_drive_powers_in(po, eff)

    def set_outputs(self, n, outputs=None):
        if outputs is None:
            outputs = {}
        outputs.update(self._outputs or {})

        for t, names in self.types.items():
            names = names - set(outputs)
            if names:
                outputs.update(zip(names, np.empty((len(names), n), dtype=t)))

        self.outputs = outputs

    def yield_results(self, gears, final_drive_speeds_out,
                      final_drive_torques_out, final_drive_powers_out):
        outputs = self.outputs
        r_gen = self.yield_ratio(gears)

        s_gen = self.yield_speed(
            final_drive_speeds_out, outputs['final_drive_ratio_vector']
        )

        t_gen = self.yield_torque(
            final_drive_torques_out, outputs['final_drive_ratio_vector']
        )

        p_gen = self.yield_power(
            final_drive_torques_out, outputs['final_drive_ratio_vector'],
            outputs['final_drive_torques_in'], final_drive_powers_out
        )

        for i, r in enumerate(r_gen):
            outputs['final_drive_ratio_vector'][i] = r
            outputs['final_drive_speeds_in'][i] = s = next(s_gen)
            l, t = next(t_gen)
            outputs['final_drive_torque_losses'][i] = l
            outputs['final_drive_torques_in'][i] = t
            e, p = next(p_gen)
            outputs['final_drive_efficiencies'][i] = e
            outputs['final_drive_powers_in'][i] = p
            yield r, s, l, t, e, p


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


def define_final_drive_prediction_model(
        final_drive_ratios, final_drive_torque_loss, n_wheel_drive,
        final_drive_efficiency):
    """
    Defines the final drive prediction model.

    :param final_drive_ratios:
        Final drive ratios [-].
    :type final_drive_ratios: dict[int | float]

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
    if final_drive_torque_loss is sh.EMPTY:
        final_drive_torque_loss = None
    model = FinalDriveModel(
        final_drive_ratios, final_drive_torque_loss, n_wheel_drive,
        final_drive_efficiency
    )
    return model


def final_drive():
    """
    Defines the final drive model.

    .. dispatcher:: d

        >>> d = final_drive()

    :return:
        The final drive model.
    :rtype: schedula.Dispatcher
    """

    d = sh.Dispatcher(
        name='Final drive',
        description='Models the final drive.'
    )

    from .defaults import dfl
    d.add_data(
        data_id='final_drive_ratio',
        default_value=dfl.values.final_drive_ratio
    )

    d.add_function(
        function=calculate_final_drive_ratios,
        inputs=['final_drive_ratio', 'n_gears'],
        outputs=['final_drive_ratios']
    )

    d.add_function(
        function=sh.add_args(calculate_final_drive_ratios),
        inputs=['gear_box_type', 'final_drive_ratio'],
        outputs=['final_drive_ratios'],
        input_domain=is_cvt
    )

    d.add_function(
        function=calculate_final_drive_ratio_vector,
        inputs=['final_drive_ratios', 'gears'],
        outputs=['final_drive_ratio_vector']
    )

    d.add_function(
        function=calculate_final_drive_speeds_in,
        inputs=['final_drive_speeds_out', 'final_drive_ratio_vector'],
        outputs=['final_drive_speeds_in']
    )

    d.add_data(
        data_id='final_drive_efficiency',
        default_value=dfl.values.final_drive_efficiency
    )

    d.add_data(
        data_id='n_wheel_drive',
        default_value=dfl.values.n_wheel_drive
    )

    d.add_data(
        data_id='final_drive_torque_loss',
        default_value=sh.EMPTY
    )

    d.add_function(
        function=calculate_final_drive_torque_losses,
        inputs=['final_drive_torques_out', 'final_drive_torque_loss'],
        outputs=['final_drive_torque_losses'],
        input_domain=lambda *args: args[1] is not sh.EMPTY
    )

    d.add_function(
        function=sh.add_args(calculate_final_drive_torque_losses_v1),
        inputs=['n_dyno_axes', 'n_wheel_drive', 'final_drive_torques_out',
                'final_drive_ratio_vector', 'final_drive_efficiency'],
        outputs=['final_drive_torque_losses'],
        weight=5,
        input_domain=domain_final_drive_torque_losses_v1
    )

    d.add_function(
        function=calculate_final_drive_torques_in,
        inputs=['final_drive_torques_out', 'final_drive_ratio_vector',
                'final_drive_torque_losses'],
        outputs=['final_drive_torques_in']
    )

    d.add_function(
        function=calculate_final_drive_efficiencies,
        inputs=['final_drive_torques_out', 'final_drive_ratio_vector',
                'final_drive_torques_in'],
        outputs=['final_drive_efficiencies']
    )

    d.add_function(
        function=calculate_final_drive_powers_in,
        inputs=['final_drive_powers_out', 'final_drive_efficiencies'],
        outputs=['final_drive_powers_in']
    )

    d.add_function(
        function=define_fake_final_drive_prediction_model,
        inputs=[
            'final_drive_ratio_vector', 'final_drive_speeds_in',
            'final_drive_torque_losses', 'final_drive_torques_in',
            'final_drive_efficiencies', 'final_drive_powers_in'
        ],
        outputs=['final_drive_prediction_model']
    )

    d.add_function(
        function=define_final_drive_prediction_model,
        inputs=[
            'final_drive_ratios', 'final_drive_torque_loss', 'n_wheel_drive',
            'final_drive_efficiency'
        ],
        outputs=['final_drive_prediction_model'],
        weight=4000
    )

    return d
