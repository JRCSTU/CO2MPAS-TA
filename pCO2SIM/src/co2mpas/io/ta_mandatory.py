import functools
import schedula as sh


def _cycle_condition(data, k):
    return sh.are_in_nested_dicts(data, *(k[:-1] + ('vehicle_mass',)))


def extend_checks(base, *extras):
    base = {k: list(v) for k, v in base.items()}
    for extra in extras:
        for k, v in extra.items():
            sh.get_nested_dicts(base, k, default=list).append(v)
    return base


def _get(d, k):
    return sh.are_in_nested_dicts(d, *k) and sh.get_nested_dicts(d, *k)


def fuel_saving_at_strategy(d, k):
    return _get(d, k[:-1] + ('gear_box_type',)) == 'automatic'


def gear_box_ratios(d, k):
    return _get(d, k[:-1] + ('gear_box_type',)) in ('automatic', 'manual')


def active_cylinder_ratios(d, k):
    return _get(d, k[:-1] + ('engine_has_cylinder_deactivation',))


def ki_multiplicative(d, k):
    if _get(d, k[:-1] + ('has_periodically_regenerating_systems',)):
        return not sh.are_in_nested_dicts(d, *(k[:-1] + ('ki_additive',)))


def ki_additive(d, k):
    if _get(d, k[:-1] + ('has_periodically_regenerating_systems',)):
        return not sh.are_in_nested_dicts(d, *(k[:-1] + ('ki_multiplicative',)))


def final_drive_ratio(d, k):
    return not sh.are_in_nested_dicts(d, *(k[:-1] + ('final_drive_ratios',)))


def final_drive_ratios(d, k):
    return not sh.are_in_nested_dicts(d, *(k[:-1] + ('final_drive_ratio',)))


def start_stop_activation_time(d, k):
    return _get(d, k[:-1] + ('has_start_stop',))


def target_gears(d, k):
    return _get(d, k[:-1] + ('gear_box_type',)) == 'manual'


base = extend_checks(
    {},
    dict.fromkeys((
        'fuel_type', 'engine_fuel_lower_heating_value', 'fuel_heating_value',
        'fuel_carbon_content_percentage', 'ignition_type', 'engine_capacity',
        'engine_stroke', 'idle_engine_speed_median', 'engine_n_cylinders',
        'engine_idle_fuel_consumption', 'tyre_code', 'gear_box_type',
        'alternator_nominal_voltage', 'alternator_nominal_power',
        'battery_capacity', 'battery_voltage', 'alternator_efficiency',
        'gear_box_ratios', 'full_load_speeds', 'full_load_powers',
        'engine_is_turbo', 'has_start_stop', 'has_energy_recuperation',
        'has_torque_converter', 'has_lean_burn',
        'has_periodically_regenerating_systems',
        'engine_has_cylinder_deactivation',
        'engine_has_variable_valve_actuation',
        'has_selective_catalytic_reduction',
        'has_gear_box_thermal_management', 'has_exhausted_gas_recirculation',
        'vehicle_mass', 'f0', 'f1', 'f2', 'n_wheel_drive', 'final_drive_ratio',
        'final_drive_ratios', 'start_stop_activation_time',
        'fuel_saving_at_strategy', 'active_cylinder_ratios',
    ), _cycle_condition),
    {
        'final_drive_ratio': final_drive_ratio,
        'final_drive_ratios': final_drive_ratios,
        'start_stop_activation_time': start_stop_activation_time,
        'fuel_saving_at_strategy': fuel_saving_at_strategy,
        'active_cylinder_ratios': active_cylinder_ratios,
        'gear_box_ratios': gear_box_ratios
    }
)

nedc = extend_checks(
    base,
    dict.fromkeys(('ki_multiplicative', 'ki_additive'), _cycle_condition),
    {
        'ki_multiplicative': ki_multiplicative,
        'ki_additive': ki_additive
    }
)

wltp = extend_checks(base, dict.fromkeys((
    'co2_emission_low', 'co2_emission_medium', 'co2_emission_high',
    'co2_emission_extra_high', 'n_dyno_axes', 'initial_temperature',
    'rcb_correction', 'speed_distance_correction',
    'times', 'velocities', 'obd_velocities', 'engine_speeds_out',
    'engine_coolant_temperatures', 'co2_normalization_references',
    'alternator_currents', 'battery_currents'
), _cycle_condition))


def _rel_cycle_cond(d, k, stage='calibration', cycle='wltp_l'):
    keys = 'base', 'input', stage, cycle, 'vehicle_mass'
    return sh.are_in_nested_dicts(d, *keys)


_mandatory = lambda *a: True

meta = (
    'times', 'velocities', 'obd_velocities', 'engine_speeds_out',
    'engine_coolant_temperatures', 'co2_normalization_references',
    'alternator_currents', 'battery_currents', 'co2_emission_low',
    'co2_emission_medium', 'co2_emission_high', 'co2_emission_extra_high',
    'rcb_correction', 'speed_distance_correction',
)


def _meta_mandatory(d, k):
    for i in ('', '.10hz', '.target'):
        if sh.are_in_nested_dicts(d, *(k[:-2] + (k[-2] + i,))):
            return True


checks = {
    'flag': {
        'vehicle_family_id': [_mandatory],
        'input_version': [_mandatory]
    },
    'dice': {
        'bifuel': [_mandatory],
        'extension': [_mandatory],
        'atct_family_correction_factor': [_mandatory]
    },
    'base': {
        'target': {
            'calibration': {
                'wltp_h': {
                    'gears': [_mandatory, target_gears],
                },
                'wltp_l': {
                    'gears': [_rel_cycle_cond, target_gears],
                },
            },
            'prediction': {
                'nedc_h': {'declared_co2_emission_value': [_mandatory]},
                'nedc_l': {'declared_co2_emission_value': [functools.partial(
                    _rel_cycle_cond, stage='prediction', cycle='nedc_l'
                )]},
                'wltp_h': {
                    'declared_co2_emission_value': [_mandatory],
                    'fuel_consumption_value': [_mandatory],
                    'corrected_co2_emission_value': [_mandatory]
                },
                'wltp_l': {
                    'declared_co2_emission_value': [_rel_cycle_cond],
                    'fuel_consumption_value': [_rel_cycle_cond],
                    'corrected_co2_emission_value': [_rel_cycle_cond]
                },
            }
        },
        'input': {
            'calibration': {
                'wltp_l': wltp,
                'wltp_h': sh.combine_dicts(wltp, {'vehicle_mass': [_mandatory]})
            },
            'prediction': {
                'nedc_l': nedc,
                'nedc_h': sh.combine_dicts(nedc, {'vehicle_mass': [_mandatory]})
            }
        }
    },
    'meta': {
        'wltp_h.10hz': dict.fromkeys(('times', 'velocities'), [_mandatory]),
        'wltp_l.10hz': dict.fromkeys(('times', 'velocities'),
                                     [_rel_cycle_cond]),
        'wltp_h.test_b': dict.fromkeys(meta, [_meta_mandatory]),
        'wltp_h.test_b.target': dict.fromkeys(
            ('fuel_consumption_value', 'corrected_co2_emission_value'),
            [_meta_mandatory]
        ),
        'wltp_h.test_b.10hz': dict.fromkeys(
            ('times', 'velocities'), [_meta_mandatory]
        ),
        'wltp_l.test_b': dict.fromkeys(meta, [_meta_mandatory]),
        'wltp_l.test_b.target': dict.fromkeys(
            ('fuel_consumption_value', 'corrected_co2_emission_value'),
            [_meta_mandatory]
        ),
        'wltp_l.test_b.10hz': dict.fromkeys(
            ('times', 'velocities'), [_meta_mandatory]
        ),
        'wltp_h.test_c': dict.fromkeys(meta, [_meta_mandatory]),
        'wltp_h.test_c.target': dict.fromkeys(
            ('fuel_consumption_value', 'corrected_co2_emission_value'),
            [_meta_mandatory]
        ),
        'wltp_h.test_c.10hz': dict.fromkeys(
            ('times', 'velocities'), [_meta_mandatory]
        ),
        'wltp_l.test_c': dict.fromkeys(meta, [_meta_mandatory]),
        'wltp_l.test_c.target': dict.fromkeys(
            ('fuel_consumption_value', 'corrected_co2_emission_value'),
            [_meta_mandatory]
        ),
        'wltp_l.test_c.10hz': dict.fromkeys(
            ('times', 'velocities'), [_meta_mandatory]
        ),
    }
}


def check_mandatory_inputs(data):
    err = []
    for k, funcs in sh.stack_nested_keys(checks):
        if all(f(data, k) for f in funcs):
            not sh.are_in_nested_dicts(data, *k) and err.append(k)

    return err
