# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the electrics of the vehicle.

Sub-Modules:

.. currentmodule:: co2mpas.core.model.physical.electrics

.. autosummary::
    :nosignatures:
    :toctree: electrics/

    motors
    electrics_prediction
"""

import math
import numpy as np
import schedula as sh
from ..defaults import dfl
import co2mpas.utils as co2_utl

dsp = sh.BlueDispatcher(
    name='Electrics', description='Models the vehicle electrics.'
)

dsp.add_data('starter_efficiency', 1)#dfl.values.starter_efficiency)
dsp.add_data('delta_time_engine_starter', dfl.values.delta_time_engine_starter)


@sh.add_function(dsp, outputs=['start_demand'], weight=100)
def calculate_engine_start_demand(
        engine_moment_inertia, idle_engine_speed, starter_efficiency,
        delta_time_engine_starter):
    """
    Calculates the energy required to start the engine [kJ].

    :param engine_moment_inertia:
        Engine moment of inertia [kg*m2].
    :type engine_moment_inertia: float

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param starter_efficiency:
        Starter efficiency [-].
    :type starter_efficiency: float

    :param delta_time_engine_starter:
        Time elapsed to turn on the engine with electric starter [s].
    :type delta_time_engine_starter: float

    :return:
        Energy required to start engine [kJ].
    :rtype: float
    """

    idle = idle_engine_speed[0] / 30.0 * math.pi
    dt = delta_time_engine_starter  # Assumed time for engine turn on [s].

    return engine_moment_inertia / starter_efficiency * idle ** 2 / 2000 * dt







dsp.add_data(
    'alternator_start_window_width', dfl.values.alternator_start_window_width
)


@sh.add_function(dsp, outputs=['starts_windows'])
def identify_alternator_starts_windows(
        times, engine_starts, alternator_currents,
        alternator_start_window_width, alternator_current_threshold):
    """
    Identifies the alternator starts windows [-].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_starts:
        When the engine starts [-].
    :type engine_starts: numpy.array

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param alternator_start_window_width:
        Alternator start window width [s].
    :type alternator_start_window_width: float

    :param alternator_current_threshold:
        Alternator current threshold [A].
    :type alternator_current_threshold: float

    :return:
        Alternator starts windows [-].
    :rtype: numpy.array
    """

    starts_windows = np.zeros_like(times, dtype=bool)
    dt = alternator_start_window_width / 2
    for i, j in _starts_windows(times, engine_starts, dt):
        b = (alternator_currents[i:j] >= alternator_current_threshold).any()
        starts_windows[i:j] = b
    return starts_windows


def _compile_alternator_powers_demand(
        alternator_nominal_voltage, alternator_efficiency):
    c = - alternator_nominal_voltage / (1000.0 * alternator_efficiency)

    def _func(alternator_currents):
        return np.maximum(alternator_currents * c, 0.0)

    return _func


@sh.add_function(dsp, outputs=['alternator_powers_demand'])
def calculate_alternator_powers_demand(
        alternator_nominal_voltage, alternator_currents, alternator_efficiency):
    """
    Calculates the alternator power demand to the engine [kW].

    :param alternator_nominal_voltage:
        Alternator nominal voltage [V].
    :type alternator_nominal_voltage: float

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param alternator_efficiency:
        Alternator efficiency [-].
    :type alternator_efficiency: float

    :return:
        Alternator power demand to the engine [kW].
    :rtype: numpy.array
    """
    return _compile_alternator_powers_demand(
        alternator_nominal_voltage, alternator_efficiency
    )(alternator_currents)


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['alternator_status_model'])
def define_alternator_status_model(
        state_of_charge_balance, state_of_charge_balance_window):
    """
    Defines the alternator status model.

    :param state_of_charge_balance:
        Battery state of charge balance [%].

        .. note::

            `state_of_charge_balance` = 99 is equivalent to 99%.
    :type state_of_charge_balance: float

    :param state_of_charge_balance_window:
        Battery state of charge balance window [%].

        .. note::

            `state_of_charge_balance_window` = 2 is equivalent to 2%.
    :type state_of_charge_balance_window: float

    :return:
        A function that predicts the alternator status.
    :rtype: callable
    """

    # noinspection PyMissingOrEmptyDocstring,PyPep8Naming
    def bers_pred(X):
        return [X[0][0] < 0]

    model = AlternatorStatusModel(
        charge_pred=lambda X: [X[0][0] == 1],
        bers_pred=bers_pred,
        min_soc=state_of_charge_balance - state_of_charge_balance_window / 2,
        max_soc=state_of_charge_balance + state_of_charge_balance_window / 2
    )

    return model


dsp.add_data('has_energy_recuperation', dfl.values.has_energy_recuperation)


# noinspection PyMissingOrEmptyDocstring
class ElectricModel(co2_utl.BaseModel):
    key_outputs = (
        'alternator_currents', 'alternator_statuses',
        'alternator_powers_demand', 'battery_currents', 'state_of_charges'
    )
    contract_outputs = 'state_of_charges',
    types = {
        float: {
            'alternator_currents', 'battery_currents', 'state_of_charges',
            'alternator_powers_demand'
        },
        int: {'alternator_statuses'}
    }

    def __init__(self, battery_capacity=None, alternator_status_model=None,
                 max_alternator_current=None, alternator_current_model=None,
                 max_battery_charging_current=None,
                 alternator_nominal_voltage=None,
                 start_demand=None, electric_load=None,
                 has_energy_recuperation=None,
                 alternator_initialization_time=None,
                 initial_state_of_charge=None, alternator_efficiency=None,
                 outputs=None):
        self.battery_capacity = battery_capacity
        self.alternator_status_model = alternator_status_model
        self.max_alternator_current = max_alternator_current
        self.alternator_current_model = alternator_current_model
        self.max_battery_charging_current = max_battery_charging_current
        self.alternator_nominal_voltage = alternator_nominal_voltage
        self.start_demand = start_demand
        self.electric_load = electric_load
        self.has_energy_recuperation = has_energy_recuperation
        self.alternator_initialization_time = alternator_initialization_time
        self.initial_state_of_charge = initial_state_of_charge
        self.alternator_efficiency = alternator_efficiency
        super(ElectricModel, self).__init__(outputs)

    def init_alternator(self, times, accelerations, gear_box_powers_in,
                        on_engine, engine_starts, state_of_charges,
                        alternator_statuses):
        keys = ['alternator_statuses', 'alternator_currents']
        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            statuses = self._outputs['alternator_statuses']
            currents = self._outputs['alternator_currents']
            _next = lambda i: (statuses[i], currents[i])
        else:
            import functools
            from .electrics_prediction import (
                predict_alternator_status, calculate_engine_start_current,
                calculate_alternator_current
            )
            init_time = self.alternator_initialization_time
            try:
                init_time += times[0]
            except IndexError:
                pass
            alt_st_mdl = functools.partial(
                self.alternator_status_model, self.has_energy_recuperation,
                init_time
            )
            self.outputs['alternator_statuses'][0] = 0

            def _next(i):
                gbp, on_eng, acc, t, eng_st, soc = (
                    gear_box_powers_in[i], on_engine[i], accelerations[i],
                    times[i], engine_starts[i], state_of_charges[i]
                )
                j = i + 1
                dt = len(times) > j and times[j] - times[i] or 0

                prev_status = i != 0 and alternator_statuses[i - 1] or 0
                alt_status = predict_alternator_status(
                    alt_st_mdl, t, prev_status, soc, gbp
                )

                sc = calculate_engine_start_current(
                    eng_st, self.start_demand, self.alternator_nominal_voltage,
                    dt
                )

                alt_current = calculate_alternator_current(
                    alt_status, on_eng, gbp, self.max_alternator_current,
                    self.alternator_current_model, sc, soc, acc, t
                )
                return alt_status, alt_current

        return _next

    def init_battery(self, times, on_engine, alternator_currents,
                     state_of_charges, battery_currents):
        keys = ['state_of_charges', 'battery_currents']

        if self._outputs is not None and not (set(keys) - set(self._outputs)):
            socs = self._outputs['state_of_charges']
            currents = self._outputs['battery_currents']
            n = len(socs) - 1
            _next = lambda i: (socs[min(i + 1, n)], currents[i])
        else:
            self.outputs['state_of_charges'][0] = self.initial_state_of_charge
            from .electrics_prediction import (
                calculate_battery_current, calculate_battery_state_of_charge
            )

            def _next(i):
                j = i + 1
                dt = len(times) > j and times[j] - times[i] or 0
                ac, on_eng = alternator_currents[i], on_engine[i]
                bc = calculate_battery_current(
                    self.electric_load, ac, self.alternator_nominal_voltage,
                    on_eng, self.max_battery_charging_current
                )

                soc = calculate_battery_state_of_charge(
                    state_of_charges[i], self.battery_capacity, dt, bc,
                    None if i == 0 else battery_currents[i - 1]
                )
                return soc, bc

        return _next

    def init_power(self, alternator_currents):
        key = 'alternator_powers_demand'
        if self._outputs is not None and key in self._outputs:
            out = self._outputs[key]
            return lambda i: out[i]

        func = _compile_alternator_powers_demand(
            self.alternator_nominal_voltage, self.alternator_efficiency
        )

        def _next(i):
            return func(alternator_currents[i])

        return _next

    def init_results(self, times, accelerations, on_engine, engine_starts,
                     gear_box_powers_in):
        outputs = self.outputs
        socs, sts = outputs['state_of_charges'], outputs['alternator_statuses']
        alt, bat = outputs['alternator_currents'], outputs['battery_currents']
        pwr = outputs['alternator_powers_demand']

        a_gen = self.init_alternator(
            times, accelerations, gear_box_powers_in, on_engine, engine_starts,
            socs, sts
        )
        b_gen = self.init_battery(times, on_engine, alt, socs, bat)
        p_gen = self.init_power(alt)

        def _next(i):
            sts[i], alt[i] = alt_status, alt_current = a_gen(i)
            pwr[i] = p = p_gen(i)
            soc, bat_current = b_gen(i)
            bat[i] = bat_current
            try:
                socs[i + 1] = soc
            except IndexError:
                pass
            return alt_current, alt_status, p, bat_current, socs[i]

        return _next


@sh.add_function(dsp, outputs=['electrics_prediction_model'], weight=4000)
def define_electrics_prediction_model(
        battery_capacity, alternator_status_model, max_alternator_current,
        alternator_current_model, max_battery_charging_current,
        alternator_nominal_voltage, start_demand, electric_load,
        has_energy_recuperation, alternator_initialization_time,
        initial_state_of_charge, alternator_efficiency):
    """
    Defines the electrics prediction model.

    :param battery_capacity:
        Battery capacity [Ah].
    :type battery_capacity: float

    :param alternator_status_model:
        A function that predicts the alternator status.
    :type alternator_status_model: AlternatorStatusModel

    :param max_alternator_current:
        Max feasible alternator current [A].
    :type max_alternator_current: float

    :param alternator_current_model:
        Alternator current model.
    :type alternator_current_model: callable

    :param max_battery_charging_current:
        Maximum charging current of the battery [A].
    :type max_battery_charging_current: float

    :param alternator_nominal_voltage:
        Alternator nominal voltage [V].
    :type alternator_nominal_voltage: float

    :param start_demand:
         Energy required to start engine [kJ].
    :type start_demand: float

    :param electric_load:
        Vehicle electric load (engine off and on) [kW].
    :type electric_load: (float, float)

    :param has_energy_recuperation:
        Does the vehicle have energy recuperation features?
    :type has_energy_recuperation: bool

    :param alternator_initialization_time:
        Alternator initialization time delta [s].
    :type alternator_initialization_time: float

    :param initial_state_of_charge:
        Initial state of charge of the battery [%].

        .. note::

            `initial_state_of_charge` = 99 is equivalent to 99%.
    :type initial_state_of_charge: float

    :param alternator_efficiency:
        Alternator efficiency [-].
    :type alternator_efficiency: float

    :return:
       Electrics prediction model.
    :rtype: ElectricModel
    """

    model = ElectricModel(
        battery_capacity, alternator_status_model, max_alternator_current,
        alternator_current_model, max_battery_charging_current,
        alternator_nominal_voltage, start_demand, electric_load,
        has_energy_recuperation, alternator_initialization_time,
        initial_state_of_charge, alternator_efficiency
    )

    return model


@sh.add_function(dsp, outputs=['electrics_prediction_model'])
def define_fake_electrics_prediction_model(
        alternator_currents, alternator_statuses, battery_currents,
        state_of_charges, alternator_powers_demand):
    """
    Defines a fake electrics prediction model.

    :param alternator_currents:
        Alternator current vector [A].
    :type alternator_currents: numpy.array

    :param alternator_statuses:
        The alternator status (0: off, 1: on, due to state of charge, 2: on due
        to BERS, 3: on and initialize battery) [-].
    :type alternator_statuses: numpy.array

    :param battery_currents:
        Low voltage battery current vector [A].
    :type battery_currents: numpy.array

    :param state_of_charges:
        State of charge of the battery [%].

        .. note::

            `state_of_charges` = 99 is equivalent to 99%.
    :type state_of_charges: numpy.array

    :param alternator_powers_demand:
        Alternator power demand to the engine [kW].
    :type alternator_powers_demand: numpy.array, optional

    :return:
       Electrics prediction model.
    :rtype: ElectricModel
    """

    model = ElectricModel(outputs={
        'alternator_currents': alternator_currents,
        'alternator_statuses': alternator_statuses,
        'battery_currents': battery_currents,
        'state_of_charges': state_of_charges,
        'alternator_powers_demand': alternator_powers_demand
    })
    return model


@sh.add_function(
    dsp,
    outputs=['alternator_currents', 'alternator_statuses', 'battery_currents',
             'state_of_charges']
)
def predict_vehicle_electrics(
        electrics_prediction_model, times, gear_box_powers_in, on_engine,
        engine_starts, accelerations):
    """
    Predicts alternator and battery currents, state of charge, and alternator
    status.

    :param electrics_prediction_model:
        Electrics prediction model.
    :type electrics_prediction_model: ElectricModel

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param gear_box_powers_in:
        Gear box power vector [kW].
    :type gear_box_powers_in: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param engine_starts:
        When the engine starts [-].
    :type engine_starts: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :return:
        Alternator and battery currents, state of charge, and alternator status
        [A, A, %, -].
    :rtype: (numpy.array, numpy.array, numpy.array, numpy.array)
    """
    return electrics_prediction_model(
        times, accelerations, on_engine, engine_starts, gear_box_powers_in
    )


dsp.add_function(
    function_id='identify_alternator_nominal_power',
    function=lambda x: max(x),
    inputs=['alternator_powers_demand'],
    outputs=['alternator_nominal_power']
)

