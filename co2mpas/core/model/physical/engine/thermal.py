# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the engine coolant temperature.
"""
import numpy as np
import schedula as sh
from ..defaults import dfl

dsp = sh.BlueDispatcher(
    name='thermal', description='Models the engine thermal behaviour.'
)


@sh.add_function(dsp, outputs=['engine_temperature_derivatives'])
def calculate_engine_temperature_derivatives(
        times, engine_coolant_temperatures):
    """
    Calculates the derivative of the engine temperature [°C/s].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Derivative of the engine temperature [°C/s].
    :rtype: numpy.array
    """

    import scipy.misc as sci_misc
    import scipy.interpolate as sci_itp
    par = dfl.functions.calculate_engine_temperature_derivatives

    func = sci_itp.InterpolatedUnivariateSpline(
        times, engine_coolant_temperatures, k=1
    )

    return sci_misc.derivative(func, times, dx=par.dx, order=par.order)


@sh.add_function(dsp, outputs=['max_engine_coolant_temperature'])
def identify_max_engine_coolant_temperature(engine_coolant_temperatures):
    """
    Identifies maximum engine coolant temperature [°C].

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Maximum engine coolant temperature [°C].
    :rtype: float
    """

    return engine_coolant_temperatures.max()


@sh.add_function(dsp, outputs=['engine_temperature_regression_model'])
def calibrate_engine_temperature_regression_model(
        engine_thermostat_temperature, on_engine,
        engine_temperature_derivatives, engine_coolant_temperatures,
        gross_engine_powers_out, engine_speeds_out_hot, accelerations):
    """
    Calibrates an engine temperature regression model to predict engine
    temperatures.

    :param engine_thermostat_temperature:
        Engine thermostat temperature [°C].
    :type engine_thermostat_temperature: float

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param engine_temperature_derivatives:
        Derivative of the engine temperature [°C/s].
    :type engine_temperature_derivatives: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param gross_engine_powers_out:
        Gross engine power (pre-losses) [kW].
    :type gross_engine_powers_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :return:
        The calibrated engine temperature regression model.
    :rtype: callable
    """
    from ._thermal import ThermalModel
    return ThermalModel(engine_thermostat_temperature).fit(
        on_engine, engine_temperature_derivatives,
        engine_coolant_temperatures, gross_engine_powers_out,
        engine_speeds_out_hot, accelerations
    )


@sh.add_function(dsp, outputs=['engine_coolant_temperatures'])
def predict_engine_coolant_temperatures(
        engine_temperature_regression_model, times, gross_engine_powers_out,
        engine_speeds_out_hot, accelerations, initial_engine_temperature,
        max_engine_coolant_temperature):
    """
    Predicts the engine temperature [°C].

    :param engine_temperature_regression_model:
        Engine temperature regression engine_temperature_regression_model.
    :type engine_temperature_regression_model: callable

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :param gross_engine_powers_out:
        Gross engine power (pre-losses) [kW].
    :type gross_engine_powers_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param initial_engine_temperature:
        Engine initial temperature [°C]
    :type initial_engine_temperature: float

    :param max_engine_coolant_temperature:
        Maximum engine coolant temperature [°C].
    :type max_engine_coolant_temperature: float

    :return:
        Engine coolant temperature vector [°C].
    :rtype: numpy.array
    """

    temp = engine_temperature_regression_model(
        np.diff(times), gross_engine_powers_out, engine_speeds_out_hot,
        accelerations, initial_temperature=initial_engine_temperature,
        max_temp=max_engine_coolant_temperature
    )

    return temp


# noinspection PyPep8Naming
@sh.add_function(dsp, outputs=['engine_thermostat_temperature'])
def identify_engine_thermostat_temperature(
        idle_engine_speed, engine_temperature_derivatives, accelerations,
        engine_coolant_temperatures, gear_box_powers_out,
        engine_speeds_out_hot):
    """
    Identifies thermostat engine temperature and its limits [°C].

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :param engine_temperature_derivatives:
        Derivative of the engine temperature [°C/s].
    :type engine_temperature_derivatives: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param gear_box_powers_out:
        Gear box power out vector [kW].
    :type gear_box_powers_out: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :param accelerations:
        Acceleration vector [m/s2].
    :type accelerations: numpy.array

    :return:
        Engine thermostat temperature [°C].
    :rtype: float
    """
    import xgboost as xgb
    from ._thermal import _build_samples
    X, Y = _build_samples(
        engine_temperature_derivatives, engine_coolant_temperatures,
        gear_box_powers_out, engine_speeds_out_hot, accelerations
    )
    X, Y = np.column_stack((Y, X[:, 1:])), X[:, 0]
    t_max, t_min = Y.max(), Y.min()
    b = (t_max - (t_max - t_min) / 3) <= Y

    model = xgb.XGBRegressor()
    model.fit(X[b], Y[b])
    ratio = np.arange(1, 1.5, 0.1) * idle_engine_speed[0]
    spl = np.zeros((len(ratio), 4))
    spl[:, 2] = ratio
    # noinspection PyTypeChecker
    return float(np.median(model.predict(spl)))


@sh.add_function(dsp, outputs=['engine_thermostat_temperature_window'])
def identify_engine_thermostat_temperature_window(
        engine_thermostat_temperature, engine_coolant_temperatures):
    """
    Identifies thermostat engine temperature limits [°C].

    :param engine_thermostat_temperature:
        Engine thermostat temperature [°C].
    :type engine_thermostat_temperature: float

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Thermostat engine temperature limits [°C].
    :rtype: float, float
    """

    thr = engine_thermostat_temperature
    # noinspection PyTypeChecker
    std = np.sqrt(np.mean((engine_coolant_temperatures - thr) ** 2))
    return thr - std, thr + std


@sh.add_function(dsp, outputs=['initial_engine_temperature'])
def identify_initial_engine_temperature(engine_coolant_temperatures):
    """
    Identifies initial engine temperature [°C].

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Initial engine temperature [°C].
    :rtype: float
    """

    return float(engine_coolant_temperatures[0])


@sh.add_function(dsp, outputs=['engine_temperature_prediction_model'])
def define_fake_engine_temperature_prediction_model(
        engine_coolant_temperatures):
    """
    Defines a fake engine temperature prediction model.

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :return:
        Engine temperature prediction model.
    :rtype: EngineTemperatureModel
    """
    from ._thermal import EngineTemperatureModel
    model = EngineTemperatureModel(outputs={
        'engine_coolant_temperatures': engine_coolant_temperatures
    })

    return model


@sh.add_function(
    dsp, outputs=['engine_temperature_prediction_model'], weight=4000
)
def define_engine_temperature_prediction_model(
        initial_engine_temperature, engine_temperature_regression_model,
        max_engine_coolant_temperature):
    """
    Defines the engine temperature prediction model.

    :param initial_engine_temperature:
        Initial engine temperature [°C].
    :type initial_engine_temperature: float

    :param engine_temperature_regression_model:
        The calibrated engine temperature regression model.
    :type engine_temperature_regression_model: ThermalModel

    :param max_engine_coolant_temperature:
        Maximum engine coolant temperature [°C].
    :type max_engine_coolant_temperature: float

    :return:
        Engine temperature prediction model.
    :rtype: EngineTemperatureModel
    """
    from ._thermal import EngineTemperatureModel
    model = EngineTemperatureModel(
        initial_engine_temperature, engine_temperature_regression_model,
        max_engine_coolant_temperature
    )

    return model
