# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It provides CO2MPAS architecture model `dsp`.

Sub-Modules:

.. currentmodule:: co2mpas.core.model

.. autosummary::
    :nosignatures:
    :toctree: model/

    physical
    selector
"""

import schedula as sh
from .physical import dsp as _physical
from .selector import dsp as _selector, calibration_cycles, prediction_cycles

dsp = sh.BlueDispatcher(
    name='CO2MPAS model',
    description='Calibrates the models with WLTP data and predicts NEDC cycle.'
)

_prediction_data = [
    'angle_slope', 'alternator_nominal_voltage', 'alternator_efficiency',
    'battery_capacity', 'cycle_type', 'cycle_name', 'engine_capacity',
    'engine_stroke', 'final_drive_efficiency', 'final_drive_ratios',
    'frontal_area', 'final_drive_ratio', 'engine_thermostat_temperature',
    'aerodynamic_drag_coefficient', 'fuel_type', 'ignition_type',
    'gear_box_type', 'engine_max_power', 'engine_speed_at_max_power',
    'rolling_resistance_coeff', 'time_cold_hot_transition',
    'engine_idle_fuel_consumption', 'engine_type', 'engine_is_turbo',
    'engine_fuel_lower_heating_value', 'has_start_stop',
    'has_energy_recuperation', 'fuel_carbon_content_percentage',
    'f0', 'f1', 'f2', 'vehicle_mass', 'full_load_speeds',
    'plateau_acceleration', 'full_load_powers', 'fuel_saving_at_strategy',
    'stand_still_torque_ratio', 'lockup_speed_ratio',
    'change_gear_window_width', 'alternator_start_window_width',
    'stop_velocity', 'min_time_engine_on_after_start',
    'min_engine_on_speed', 'max_velocity_full_load_correction',
    'is_hybrid', 'tyre_code', 'engine_has_cylinder_deactivation',
    'active_cylinder_ratios', 'engine_has_variable_valve_actuation',
    'has_torque_converter', 'has_gear_box_thermal_management',
    'has_lean_burn', 'ki_additive', 'ki_multiplicative', 'n_wheel_drive',
    'has_periodically_regenerating_systems', 'n_dyno_axes',
    'has_selective_catalytic_reduction', 'has_exhausted_gas_recirculation',
    'start_stop_activation_time', 'engine_n_cylinders',
    'initial_drive_battery_state_of_charge',
    'motor_p0_speed_ratio', 'motor_p1_speed_ratio',
    'motor_p2_speed_ratio', 'motor_p2_planetary_speed_ratio',
    'motor_p3_front_speed_ratio', 'motor_p3_rear_speed_ratio',
    'motor_p4_front_speed_ratio', 'motor_p4_rear_speed_ratio',
    'rcb_correction', 'speed_distance_correction',
    'atct_family_correction_factor', 'is_plugin'
]

_prediction_data_ts = ['times', 'velocities', 'gears']

_physical = sh.SubDispatch(_physical)

dsp.add_data(
    data_id='input.calibration.wltp_h',
    description='User input data of WLTP-H calibration stage.'
)
dsp.add_data(
    data_id='input.calibration.wltp_l',
    description='User input data of WLTP-L calibration stage.'
)
dsp.add_data(
    data_id='output.calibration.wltp_h',
    description='Output data of WLTP-H calibration stage.'
)
dsp.add_data(
    data_id='output.calibration.wltp_l',
    description='Output data of WLTP-L calibration stage.'
)
dsp.add_function(
    function_id='calibrate_with_wltp_h',
    function=_physical,
    inputs=['input.calibration.wltp_h'],
    outputs=['output.calibration.wltp_h'],
    description='Wraps all functions needed to calibrate the models to '
                'predict CO2 emissions.'
)

dsp.add_function(
    function_id='calibrate_with_wltp_l',
    function=_physical,
    inputs=['input.calibration.wltp_l'],
    outputs=['output.calibration.wltp_l'],
    description='Wraps all functions needed to calibrate the models to '
                'predict CO2 emissions.'
)

dsp.add_data(
    data_id='input.prediction.wltp_h', default_value={},
    description='User input data of WLTP-H prediction stage.'
)
dsp.add_data(
    data_id='input.prediction.wltp_l', default_value={},
    description='User input data of WLTP-L prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.models_wltp_h',
    description='Calibrated models for WLTP-H prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.models_wltp_l',
    description='Calibrated models for WLTP-L prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.wltp_h',
    description='Input data of WLTP-H prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.wltp_l',
    description='Input data of WLTP-L prediction stage.'
)


@sh.add_function(
    dsp, inputs=['output.calibration.wltp_h', 'data.prediction.models_wltp_h',
                 'input.prediction.wltp_h'], outputs=['data.prediction.wltp_h']
)
@sh.add_function(
    dsp, inputs=['output.calibration.wltp_l', 'data.prediction.models_wltp_l',
                 'input.prediction.wltp_l'], outputs=['data.prediction.wltp_l']
)
def select_prediction_data(data, *new_data):
    """
    Selects the data required to predict the CO2 emissions with CO2MPAS model.

    :param data:
        Output data.
    :type data: dict

    :param new_data:
        New data.
    :type new_data: dict

    :return:
        Data required to predict the CO2 emissions with CO2MPAS model.
    :rtype: dict
    """

    ids = _prediction_data
    from co2mpas.defaults import dfl
    if not dfl.functions.select_prediction_data.theoretical:
        ids = ids + _prediction_data_ts

    data = sh.selector(ids, data, allow_miss=True)

    if new_data:
        new_data = sh.combine_dicts(*new_data)
        data = sh.combine_dicts(data, new_data)

    if 'gears' in data and 'gears' not in new_data:
        if data.get('gear_box_type', 0) == 'automatic' or \
                len(data.get('velocities', ())) != len(data['gears']):
            data.pop('gears')

    return data


dsp.add_data(
    data_id='input.prediction.models', default_value={},
    description='User input models for prediction stages.'
)
dsp.add_data(
    data_id='enable_selector', default_value=False,
    description='Enable the selection of the best model to predict both '
                'H/L cycles.'
)
dsp.add_data(
    data_id='data.calibration.model_scores',
    description='Scores of calibrated models.'
)

_prediction_models = tuple(map('models_{}'.format, prediction_cycles))
dsp.add_function(
    function=sh.SubDispatchPipe(
        _selector,
        function_id='extract_calibrated_models',
        inputs=('enable_selector', 'default_models',) + calibration_cycles,
        outputs=('selections',) + _prediction_models
    ),
    inputs=[
        'enable_selector', 'input.prediction.models',
        'output.calibration.wltp_h', 'output.calibration.wltp_l'
    ],
    outputs=['data.calibration.model_scores'] + [
        'data.prediction.%s' % k for k in _prediction_models
    ]
)

dsp.add_data(
    data_id='output.prediction.wltp_h',
    description='Output data of WLTP-H prediction stage.'
)
dsp.add_data(
    data_id='output.prediction.wltp_l',
    description='Output data of WLTP-L prediction stage.'
)
dsp.add_function(
    function_id='predict_wltp_h',
    function=_physical,
    inputs=['data.prediction.wltp_h'],
    outputs=['output.prediction.wltp_h'],
    description='Wraps all functions needed to predict CO2 emissions.'
)

dsp.add_function(
    function_id='predict_wltp_l',
    function=_physical,
    inputs=['data.prediction.wltp_l'],
    outputs=['output.prediction.wltp_l'],
    description='Wraps all functions needed to predict CO2 emissions.'

)

dsp.add_data(
    data_id='input.prediction.nedc_h',
    description='User input data of NEDC-H prediction stage.'
)
dsp.add_data(
    data_id='input.prediction.nedc_l',
    description='User input data of NEDC-L prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.models_nedc_h',
    description='Calibrated models for NEDC-H prediction stage.'
)
dsp.add_data(
    data_id='data.prediction.models_nedc_l',
    description='Calibrated models for NEDC-L prediction stage.'
)
dsp.add_data(
    data_id='output.prediction.nedc_h',
    description='Output data of NEDC-H prediction stage.'
)
dsp.add_data(
    data_id='output.prediction.nedc_l',
    description='Output data of NEDC-L prediction stage.'
)
dsp.add_function(
    function_id='predict_nedc_h',
    function=_physical,
    inputs=['data.prediction.models_nedc_h', 'input.prediction.nedc_h'],
    outputs=['output.prediction.nedc_h'],
    description='Wraps all functions needed to predict CO2 emissions.'
)

dsp.add_function(
    function_id='predict_nedc_l',
    function=_physical,
    inputs=['data.prediction.models_nedc_l', 'input.prediction.nedc_l'],
    outputs=['output.prediction.nedc_l'],
    description='Wraps all functions needed to predict CO2 emissions.'
)
