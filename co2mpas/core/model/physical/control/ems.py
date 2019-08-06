# -*- coding: utf-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to model the Energy Management System.
"""
import numpy as np
import schedula as sh
from ..defaults import dfl

dsp = sh.BlueDispatcher(
    name='ems',
    description='Models the Energy Management System.'
)

dsp.add_function(
    function_id='define_motors_efficiencies',
    function=sh.bypass,
    inputs=[
        'motor_p4_efficiency', 'motor_p3_efficiency', 'motor_p2_efficiency',
        'motor_p1_efficiency', 'motor_p0_efficiency'
    ],
    outputs=['motors_efficiencies']
)

dsp.add_function(
    function_id='define_motors_maximum_powers',
    function=sh.bypass,
    inputs=[
        'motor_p4_maximum_power', 'motor_p3_maximum_power',
        'motor_p2_maximum_power', 'motor_p1_maximum_power',
        'motor_p0_maximum_power'
    ],
    outputs=['motors_maximum_powers']
)


@sh.add_function(dsp, outputs=['motors_maximums_powers'])
def define_motors_maximums_powers(
        motor_p4_maximum_powers, motor_p3_maximum_powers,
        motor_p2_maximum_powers, hypothetical_engine_speeds,
        motor_p1_maximum_power_function, motor_p1_speed_ratio,
        motor_p0_maximum_power_function, motor_p0_speed_ratio):
    """
    Defines maximum powers of electric motors [kW].

    :param motor_p4_maximum_powers:
        Maximum power vector of motor P4 [kW].
    :type motor_p4_maximum_powers: numpy.array

    :param motor_p3_maximum_powers:
        Maximum power vector of motor P3 [kW].
    :type motor_p3_maximum_powers: numpy.array

    :param motor_p2_maximum_powers:
        Maximum power vector of motor P2 [kW].
    :type motor_p2_maximum_powers: numpy.array

    :param hypothetical_engine_speeds:
        Hypothetical engine speed [RPM].
    :type hypothetical_engine_speeds: numpy.array

    :param motor_p1_maximum_power_function:
        Maximum power function of motor P1.
    :type motor_p1_maximum_power_function: function

    :param motor_p1_speed_ratio:
        Ratio between motor P1 speed and engine speed [-].
    :type motor_p1_speed_ratio: float

    :param motor_p0_maximum_power_function:
        Maximum power function of motor P0.
    :type motor_p0_maximum_power_function: function

    :param motor_p0_speed_ratio:
        Ratio between motor P0 speed and engine speed [-].
    :type motor_p0_speed_ratio: float

    :return:
        Maximum powers of electric motors [kW].
    :rtype: numpy.array
    """
    es, p2_powers = hypothetical_engine_speeds, motor_p2_maximum_powers
    return np.column_stack((
        motor_p4_maximum_powers, motor_p3_maximum_powers, p2_powers,
        motor_p1_maximum_power_function(es * motor_p1_speed_ratio),
        motor_p0_maximum_power_function(es * motor_p0_speed_ratio)
    ))


@sh.add_function(dsp, outputs=['drive_line_efficiencies'])
def define_drive_line_efficiencies(
        final_drive_mean_efficiency, gear_box_mean_efficiency,
        clutch_tc_mean_efficiency, belt_mean_efficiency):
    """
    Defines drive line efficiencies vector.

    :param final_drive_mean_efficiency:
        Final drive mean efficiency [-].
    :type final_drive_mean_efficiency: float

    :param gear_box_mean_efficiency:
        Gear box mean efficiency [-].
    :type gear_box_mean_efficiency: float

    :param clutch_tc_mean_efficiency:
        Clutch or torque converter mean efficiency [-].
    :type clutch_tc_mean_efficiency: float

    :param belt_mean_efficiency:
        Belt mean efficiency [-].
    :type belt_mean_efficiency: float

    :return:
        Drive line efficiencies vector.
    :rtype: tuple[float]
    """
    return (
        1.0, final_drive_mean_efficiency, gear_box_mean_efficiency,
        clutch_tc_mean_efficiency, belt_mean_efficiency
    )


@sh.add_function(dsp, outputs=['hypothetical_engine_speeds'])
def calculate_hypothetical_engine_speeds(gear_box_speeds_in, idle_engine_speed):
    """
    Calculate hypothetical engine speed [RPM].

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array | float

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        Hypothetical engine speed [RPM].
    :rtype: numpy.array | float
    """
    return np.maximum(gear_box_speeds_in, idle_engine_speed[0])


def _invert(y, xp, fp):
    x = xp[:-1] + (np.diff(xp) / np.diff(fp)) * (y[:, None] - fp[:-1])
    b = (xp[:-1] - dfl.EPS <= x) & (x <= (xp[1:] + dfl.EPS))
    return x[np.arange(b.shape[0]), np.nanargmax(b, 1)], y


# noinspection PyMissingOrEmptyDocstring
class FuelMapModel:
    def __init__(self, fuel_map, full_load_curve):
        from scipy.interpolate import UnivariateSpline as Spl
        from scipy.interpolate import RegularGridInterpolator as Interpolator
        self.full_load_curve = full_load_curve
        self.fc = Interpolator(
            (fuel_map['speed'], fuel_map['power']), fuel_map['fuel'],
            bounds_error=False, fill_value=0
        )
        (s, p), fc = self.fc.grid, self.fc.values

        with np.errstate(divide='ignore', invalid='ignore'):
            e = np.maximum(0, p / fc)
        e[(p > full_load_curve(s)[:, None]) | (p < 0)] = np.nan
        b = ~np.isnan(e).all(0)
        s, i = np.unique(s, return_index=True)
        p = p[b][np.nanargmax(e[:, b], 1)][i]

        func = Spl(s, p, w=1 / np.clip(p * .01, dfl.EPS, 1))
        s = np.unique(np.append(s, np.linspace(s.min(), s.max(), 1000)))
        p = func(s)
        self.max_power = p.max()
        self.speed_power = Spl(s, p, s=0)
        self.power_speed = Spl(*_invert(np.unique(p), s, p)[::-1], s=0, ext=3)
        self.idle_fc = self.fc((self.power_speed(0), 0))

    def __call__(self, speed, power):
        return self.fc((speed, power))


@sh.add_function(dsp, outputs=['fuel_map_model'])
def define_fuel_map_model(fuel_map, full_load_curve):
    """
    Define the fuel map model.

    :param fuel_map:
        Fuel consumption map [RPM, kW, g/s].
    :type fuel_map: dict

    :param full_load_curve:
        Vehicle full load curve.
    :type full_load_curve: function

    :return:
        Fuel map model.
    :rtype: FuelMapModel
    """
    return FuelMapModel(fuel_map, full_load_curve)


def _interp(x, xp, fp):
    x = np.asarray(x).clip(xp[:, 0, None], xp[:, -1, None])
    j = np.maximum(1, np.argmax(x[:, :, None] <= xp[:, None, :], axis=-1))
    i, k = j - 1, np.arange(xp.shape[0])[:, None]
    (x0, dx), (y0, dy) = xp[k, [i, j]], fp[k, [i, j]]
    dx -= x0
    dy -= y0
    with np.errstate(divide='ignore', invalid='ignore'):
        return (x - x0) * np.where(np.isclose(dx, 0), 0, dy / dx) + y0, x


# noinspection PyMissingOrEmptyDocstring
class HEV:
    def __init__(self, drive_line_efficiencies, motors_efficiencies):

        self.m_eff = np.array(motors_efficiencies)
        m_dl_eff = np.multiply.accumulate(drive_line_efficiencies)
        self.ice_eff = m_dl_eff[-2]

        # Electric assist (dp < 0).
        self.m_ds = m_dl_eff / self.ice_eff
        self.i_ds = np.argsort(self.m_eff * self.m_ds)[::-1]
        self.j_ds = np.argsort(self.i_ds)

        # Charging (dp > 0).
        self.m_ch = m_ch_eff = self.m_ds.copy()
        m_ch_eff[-1] = 1 / m_ch_eff[-1]
        self.i_ch = np.argsort(m_ch_eff / self.m_eff)
        self.j_ch = np.argsort(self.i_ch)

    def delta_ice_power(self, motors_maximums_powers):
        mmp = np.asarray(motors_maximums_powers)
        n, acc = mmp.shape[-1], np.add.accumulate
        mmp = mmp.reshape((mmp.shape[:-1] or (1,)) + (n,))
        dp_ice, p_bat = np.zeros((2, 2 * n + 1) + mmp.shape[:-1])
        dp_ice[:n] = -acc((mmp * self.m_ds).take(self.i_ds, axis=-1).T)[::-1]
        dp_ice[-n:] = acc((mmp * self.m_ch).take(self.i_ch, axis=-1).T)
        p_bat[:n] = -acc((mmp / self.m_eff).take(self.i_ds, axis=-1).T)[::-1]
        p_bat[-n:] = acc((mmp * self.m_eff).take(self.i_ch, axis=-1).T)

        # noinspection PyUnusedLocal
        def battery_power_split(battery_powers, *args):
            p = np.asarray(battery_powers)
            p = p.reshape((p.shape[:-1] or (1,)) + p.shape[-1:])
            b = p < 0
            pm = np.where(b, -p_bat[:n + 1][::-1], p_bat[-1 - n:])
            pm = np.diff(pm + np.minimum(np.abs(p) - pm, 0), axis=0)
            pm = np.where(b, pm[self.j_ds], -pm[self.j_ch])
            return pm

        return dp_ice, p_bat, battery_power_split

    def ice_power(self, motive_power, motors_maximum_powers):
        dp_ice, p_bat, power_split = self.delta_ice_power(motors_maximum_powers)
        p_ice = dp_ice + motive_power / self.ice_eff
        return p_ice, p_bat, power_split

    def parallel(self, motive_powers, motors_maximums_powers,
                 engine_powers_out=None, ice_power_losses=0,
                 battery_power_losses=0):
        pi, pb, bps = self.ice_power(motive_powers, motors_maximums_powers)
        if ice_power_losses is not 0:
            pi += np.asarray(ice_power_losses).ravel()[None, :]
        if battery_power_losses is not 0:
            pb += np.asarray(battery_power_losses).ravel()[None, :]
        if engine_powers_out is not None:
            pb, pi = _interp(engine_powers_out, pi.T, pb.T)
        return pi, pb, bps

    def ev(self, motive_powers, motors_maximums_powers, battery_power_losses=0):
        return self.parallel(motive_powers, np.pad(
            motors_maximums_powers[:, :-2], ((0, 0), (0, 2)), 'constant'
        ), 0, battery_power_losses=battery_power_losses)

    def serial(self, motive_powers, motors_maximums_powers, engine_powers_out,
               ice_power_losses=0, battery_power_losses=0):
        pi, pb_ev, bps_ev = self.ev(motive_powers, motors_maximums_powers)

        pi, pb, bps = self.parallel(
            np.zeros_like(motive_powers), np.pad(
                motors_maximums_powers[:, -2:], ((0, 0), (3, 0)), 'constant'
            ), engine_powers_out=engine_powers_out,
            ice_power_losses=ice_power_losses,
            battery_power_losses=battery_power_losses
        )

        def battery_power_split(battery_powers):
            return bps(battery_powers - pb_ev) + bps_ev(pb_ev)

        return pi, pb + pb_ev, battery_power_split


@sh.add_function(dsp, outputs=['hev_power_model'])
def define_hev_power_model(motors_efficiencies, drive_line_efficiencies):
    """
    Define Hybrid Electric Vehicle power balance model.

    :param motors_efficiencies:
        Electric motors efficiencies vector.
    :type motors_efficiencies: tuple[float]

    :param drive_line_efficiencies:
        Drive line efficiencies vector.
    :type drive_line_efficiencies: tuple[float]

    :return:
        Hybrid Electric Vehicle power balance model.
    :rtype: HEV
    """
    return HEV(drive_line_efficiencies, motors_efficiencies)


@sh.add_function(dsp, outputs=['hybrid_modes'])
def identify_hybrid_modes(
        gear_box_speeds_in, engine_speeds_out, idle_engine_speed, on_engine):
    """
    Identify the hybrid mode status (0: EV, 1: Parallel, 2: Serial).

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param idle_engine_speed:
        Idle engine speed and its standard deviation [RPM].
    :type idle_engine_speed: (float, float)

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :return:
        Hybrid mode status (0: EV, 1: Parallel, 2: Serial).
    :rtype: numpy.array
    """
    mode = on_engine.astype(int)
    b = idle_engine_speed[0] > gear_box_speeds_in
    b |= (gear_box_speeds_in - idle_engine_speed[1]) > engine_speeds_out
    mode[on_engine & b] = 2
    return mode


@sh.add_function(dsp, outputs=['serial_motor_maximum_power_function'])
def define_serial_motor_maximum_power_function(
        motor_p1_maximum_power_function, motor_p0_maximum_power_function,
        motor_p1_speed_ratio, motor_p0_speed_ratio):
    """
    Define serial motor maximum power function.

    :param motor_p1_maximum_power_function:
        Maximum power function of motor P1.
    :type motor_p1_maximum_power_function: function

    :param motor_p1_speed_ratio:
        Ratio between motor P1 speed and engine speed [-].
    :type motor_p1_speed_ratio: float

    :param motor_p0_maximum_power_function:
        Maximum power function of motor P0.
    :type motor_p0_maximum_power_function: function

    :param motor_p0_speed_ratio:
        Ratio between motor P0 speed and engine speed [-].
    :type motor_p0_speed_ratio: float

    :return:
        Serial motor maximum power function.
    :rtype: function
    """

    # noinspection PyMissingOrEmptyDocstring
    def calculate_serial_motor_maximum_power(engine_speed):
        es = np.atleast_1d(engine_speed)
        return np.pad(
            np.column_stack((
                motor_p1_maximum_power_function(es * motor_p1_speed_ratio),
                motor_p0_maximum_power_function(es * motor_p0_speed_ratio)
            )), ((0, 0), (3, 0)), 'constant'
        )

    return calculate_serial_motor_maximum_power


@sh.add_function(dsp, outputs=['engine_power_losses_function'])
def define_engine_power_losses_function(
        engine_moment_inertia, auxiliaries_torque_loss, auxiliaries_power_loss):
    """
    Define engine power losses function.

    :param engine_moment_inertia:
        Engine moment of inertia [kg*m2].
    :type engine_moment_inertia: float

    :param auxiliaries_torque_loss:
        Constant torque loss due to engine auxiliaries [N*m].
    :type auxiliaries_torque_loss: float

    :param auxiliaries_power_loss:
        Constant power loss due to engine auxiliaries [kW].
    :type auxiliaries_power_loss: float

    :return:
        Engine power losses function.
    :rtype: function
    """
    from ..engine import (
        calculate_auxiliaries_power_losses as aux_p,
        calculate_auxiliaries_torque_losses as aux_t,
        calculate_engine_inertia_powers_losses as ine_p
    )

    # noinspection PyMissingOrEmptyDocstring
    def engine_power_losses_function(times, engine_speeds, inertia=True):
        p = 0
        if inertia:
            p = ine_p(times, engine_speeds, engine_moment_inertia)
        return p + aux_p(
            aux_t(times, auxiliaries_torque_loss), engine_speeds,
            np.ones_like(engine_speeds, bool), auxiliaries_power_loss
        )

    return engine_power_losses_function


# noinspection PyMissingOrEmptyDocstring
class EMS:
    def __init__(self, battery_model, hev_power_model, fuel_map_model,
                 serial_motor_maximum_power_function, starter_model,
                 dcdc_converter_efficiency, engine_power_losses_function,
                 s_ch=None, s_ds=None):
        self.battery_model = battery_model
        self.hev_power_model = hev_power_model
        self.fuel_map_model = fuel_map_model
        self.s_ch, self.s_ds = s_ch, s_ds
        self.battery_fuel_mode = 'current'
        self._battery_power = None
        self.starter_model = starter_model
        self.serial_motor_maximum_power = serial_motor_maximum_power_function
        self.engine_power_losses = engine_power_losses_function
        self.dcdc_converter_efficiency = dcdc_converter_efficiency

    def set_virtual(self, motors_maximum_powers):
        from scipy.interpolate import UnivariateSpline as Spline
        p, pb = self.hev_power_model.delta_ice_power(motors_maximum_powers)[:-1]
        pb, i = np.unique(-pb, return_index=True)
        self._battery_power = Spline(pb, np.abs(p.ravel()[i]), k=1, s=0)

    def fit(self, hybrid_modes, times, motive_powers, motors_maximums_powers,
            engine_powers_out, engine_speeds_out, ice_power_losses=None):
        pl = ice_power_losses
        if pl is None:
            pl = self.engine_power_losses(times, engine_speeds_out)
        hev, fc = self.hev_power_model, self.fuel_map_model
        pi = engine_powers_out[:, None] + [-dfl.EPS, dfl.EPS]
        pb = np.where(hybrid_modes[:, None] == 1, *(func(
            motive_powers, motors_maximums_powers, pi, ice_power_losses=pl
        )[1] for func in (hev.parallel, hev.serial)))
        b = hybrid_modes.astype(bool)
        pi, bc = pi[b], self.battery_model.currents(pb[b])
        s = -np.diff(fc(engine_speeds_out[b, None], pi), axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s /= np.diff(bc, axis=1)
        b = bc.mean(1) >= 0
        # noinspection PyUnresolvedReferences
        self.s_ch, self.s_ds = np.nanmedian(s[b]), np.nanmedian(s[~b])
        return self

    def battery_fuel(self, battery_currents):
        bc = battery_currents
        if self.battery_fuel_mode == 'current':
            return np.where(bc >= 0, self.s_ch, self.s_ds) * bc
        pi = self._battery_power(self.battery_model.powers(bc))
        fc = self.fuel_map_model(self.fuel_map_model.power_speed(pi), pi)
        fc -= self.fuel_map_model.idle_fc
        fc *= -np.sign(bc)
        return fc

    def electric(self, motive_powers, motors_maximums_powers,
                 battery_power_losses=0):
        res, hev = {}, self.hev_power_model
        pi, pb, bps = hev.ice_power(motive_powers, np.pad(
            motors_maximums_powers[:, :-2], ((0, 0), (0, 2)), 'constant'
        ))
        res['power_bat'], res['power_ice'] = pb, pi = _interp(0, pi.T, pb.T)
        pb += np.atleast_2d(battery_power_losses).T
        res['current_bat'] = bc = self.battery_model.currents(pb)
        res['fc_bat'] = res['fc_eq'] = self.battery_fuel(bc)
        res['fc_ice'] = res['speed_ice'] = np.zeros_like(bc)
        res['battery_power_split'] = bps
        return res

    def parallel(self, times, motive_powers, motors_maximums_powers,
                 engine_speeds_out, ice_power_losses=None,
                 battery_power_losses=0, opt=True):
        hev, fc = self.hev_power_model, self.fuel_map_model
        pi, pb, bps = hev.ice_power(motive_powers, motors_maximums_powers)
        if ice_power_losses is None:
            ice_power_losses = self.engine_power_losses(
                times, engine_speeds_out
            )
        pi += np.atleast_2d(ice_power_losses)
        ep = np.minimum(fc.full_load_curve(engine_speeds_out), pi[-1])
        ep = np.linspace(0, 1, 200) * ep[:, None]
        pb, pi = _interp(ep, pi.T, pb.T)
        es = np.atleast_2d(engine_speeds_out).T
        fc_ice = fc(es, pi)
        pb += np.atleast_2d(battery_power_losses).T
        bc = self.battery_model.currents(pb)
        fc_bat = self.battery_fuel(bc)
        fc_eq = fc_ice + fc_bat
        res = dict(
            fc_eq=fc_eq, fc_ice=fc_ice, fc_bat=fc_bat, power_bat=pb,
            power_ice=pi, current_bat=bc, speed_ice=es, battery_power_split=bps
        )
        if opt:
            i, j = np.arange(fc_eq.shape[0]), np.nanargmin(fc_eq, axis=1)
            res = self.min(res, i, j)
        return self.starter_penalties(res)

    def serial(self, times, motive_powers, motors_maximums_powers,
               engine_speeds_out=None, ice_power_losses=None,
               battery_power_losses=0, opt=True):
        hev, fc, n = self.hev_power_model, self.fuel_map_model, 200
        pi, pb, bps_ev = hev.ice_power(motive_powers, np.pad(
            motors_maximums_powers[:, :-2], ((0, 0), (0, 2)), 'constant'
        ))
        pb_ev, pl = _interp(0, pi.T, pb.T)[0].ravel(), ice_power_losses
        if engine_speeds_out is None:
            # noinspection PyProtectedMember
            es = self.fuel_map_model.speed_power._data[0]
            es = np.linspace(es[0], es[-1], n)[:, None]
            if not hasattr(pl, 'shape'):
                pl = np.ones_like(motive_powers) * (pl or 0)

            es, pl = np.meshgrid(es, pl, indexing='ij')
            if ice_power_losses is None:
                pl = self.engine_power_losses(times, es, inertia=False)
            es, pl = es.ravel()[:, None], pl.ravel()
        else:
            es = engine_speeds_out[:, None]
            if pl is None:
                pl = self.engine_power_losses(times, engine_speeds_out)
        mmp = self.serial_motor_maximum_power(es)
        pi, pb, bps = hev.ice_power(0, mmp)
        pi += np.atleast_2d(pl)
        pb, pi = _interp(fc.speed_power(es), pi.T, pb.T)
        if engine_speeds_out is None:
            pb, pi = pb.reshape(n, -1).T, pi.reshape(n, -1).T
            es = es.reshape(n, -1).T
        pb += np.atleast_2d(pb_ev + battery_power_losses).T
        fc_ice = fc(es, pi)
        bc = self.battery_model.currents(pb)
        fc_bat = self.battery_fuel(bc)
        fc_eq = fc_ice + fc_bat

        def battery_power_split(battery_powers, engine_speeds):
            return hev.ice_power(
                0, self.serial_motor_maximum_power(engine_speeds)
            )[-1](battery_powers - battery_power_losses - pb_ev) + bps_ev(pb_ev)

        res = dict(
            fc_eq=fc_eq, fc_ice=fc_ice, fc_bat=fc_bat, power_bat=pb,
            power_ice=pi, current_bat=bc, speed_ice=es,
            battery_power_split=battery_power_split
        )
        if engine_speeds_out is None and opt:
            i, j = np.arange(pi.shape[0]), np.argmax(pi / fc_ice, 1)
            res = self.min(res, i, j)
            res['speed_ice'] = res['speed_ice'][i, j, None]
        return self.starter_penalties(res)

    def starter_penalties(self, res):
        cf, eff = self.battery_model.currents, self.dcdc_converter_efficiency
        c = np.array([eff, -1 / eff]) / 4
        pb = self.starter_model(res['speed_ice'])[None, :] * c[:, None, None]
        res['power_stop'], res['power_start'] = pb
        res['current_stop'], res['current_start'] = bc = cf(pb)
        res['fc_stop'], res['fc_start'] = self.battery_fuel(bc)
        return res

    @staticmethod
    def min(res, i, j):
        res, keys = res.copy(), set(res) - {'battery_power_split', 'speed_ice'}
        for k in keys:
            res[k] = res[k][i, j, None]
        return res

    def __call__(self, times, motive_powers, motors_maximums_powers,
                 gear_box_speeds_in, idle_engine_speed, battery_power_losses=0):
        e = self.electric(motive_powers, motors_maximums_powers)
        p = self.parallel(
            times, motive_powers, motors_maximums_powers,
            np.maximum(gear_box_speeds_in, idle_engine_speed[0]),
            battery_power_losses=battery_power_losses
        )
        s = self.serial(
            times, motive_powers, motors_maximums_powers,
            battery_power_losses=battery_power_losses
        )
        c_ser = np.column_stack((s['current_start'], -s['current_stop']))
        k_ser = ((s['current_bat'] - e['current_bat'] + c_ser) / s['fc_ice']).T
        c_par = np.column_stack((p['current_start'], -p['current_stop']))
        k_par = ((p['current_bat'] - e['current_bat'] + c_par) / p['fc_ice']).T

        fc_ser = s['fc_eq'] + np.column_stack((s['fc_start'], -s['fc_stop']))
        fc_par = p['fc_eq'] + np.column_stack((p['fc_start'], -p['fc_stop']))
        # noinspection PyUnresolvedReferences
        mode = (fc_ser < fc_par).astype(int) + 1
        mode[gear_box_speeds_in < -np.diff(idle_engine_speed)] = 2
        be_serial = (e['power_ice'].ravel() > dfl.EPS) & (motive_powers > 0.01)
        mode[be_serial] = 1
        k_ref = np.choose(mode - 1, [k_par.T, k_ser.T])
        return dict(
            hybrid_modes=mode, k_serial=k_ser, k_parallel=k_par, serial=s,
            parallel=p, electric=e, k_reference=k_ref, force_on_engine=be_serial
        )


@sh.add_function(dsp, outputs=['ecms_s'])
def calibrate_ems_model(
        drive_battery_model, hev_power_model, fuel_map_model, hybrid_modes,
        serial_motor_maximum_power_function, motive_powers, starter_model,
        motors_maximums_powers, engine_powers_out, engine_speeds_out, times,
        engine_power_losses_function, dcdc_converter_efficiency):
    """
    Calibrate Energy Management Strategy model.

    :param drive_battery_model:
        Drive battery current model.
    :type drive_battery_model: DriveBatteryModel

    :param hev_power_model:
        Hybrid Electric Vehicle power balance model.
    :type hev_power_model: HEV

    :param fuel_map_model:
        Fuel map model.
    :type fuel_map_model: FuelMapModel

    :param hybrid_modes:
        Hybrid mode status (0: EV, 1: Parallel, 2: Serial).
    :type hybrid_modes: numpy.array

    :param serial_motor_maximum_power_function:
        Serial motor maximum power function.
    :type serial_motor_maximum_power_function: function

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param starter_model:
        Starter model.
    :type starter_model: StarterModel

    :param dcdc_converter_efficiency:
        DC/DC converter efficiency [-].
    :type dcdc_converter_efficiency: float

    :param motors_maximums_powers:
        Maximum powers of electric motors [kW].
    :type motors_maximums_powers: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param engine_speeds_out:
        Engine speed [RPM].
    :type engine_speeds_out: numpy.array

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_power_losses_function:
        Engine power losses function.
    :type engine_power_losses_function: function

    :return:
        Equivalent Consumption Minimization Strategy params.
    :rtype: tuple[float]
    """
    model = EMS(
        drive_battery_model, hev_power_model, fuel_map_model,
        serial_motor_maximum_power_function, starter_model,
        dcdc_converter_efficiency, engine_power_losses_function).fit(
        hybrid_modes, times, motive_powers, motors_maximums_powers,
        engine_powers_out, engine_speeds_out
    )
    return model.s_ch, model.s_ds


@sh.add_function(dsp, outputs=['ems_model'])
def define_ems_model(
        drive_battery_model, hev_power_model, fuel_map_model,
        serial_motor_maximum_power_function, starter_model,
        dcdc_converter_efficiency, engine_power_losses_function, ecms_s):
    """
    Define Energy Management Strategy model.

    :param drive_battery_model:
        Drive battery current model.
    :type drive_battery_model: DriveBatteryModel

    :param hev_power_model:
        Hybrid Electric Vehicle power balance model.
    :type hev_power_model: HEV

    :param fuel_map_model:
        Fuel map model.
    :type fuel_map_model: FuelMapModel

    :param serial_motor_maximum_power_function:
        Serial motor maximum power function.
    :type serial_motor_maximum_power_function: function

    :param starter_model:
        Starter model.
    :type starter_model: StarterModel

    :param dcdc_converter_efficiency:
        DC/DC converter efficiency [-].
    :type dcdc_converter_efficiency: float

    :param engine_power_losses_function:
        Engine power losses function.
    :type engine_power_losses_function: function

    :param ecms_s:
        Equivalent Consumption Minimization Strategy params.
    :type ecms_s: tuple[float]

    :return:
        Energy Management Strategy model.
    :rtype: EMS
    """
    return EMS(
        drive_battery_model, hev_power_model, fuel_map_model,
        serial_motor_maximum_power_function, starter_model,
        dcdc_converter_efficiency, engine_power_losses_function, s_ch=ecms_s[0],
        s_ds=ecms_s[1]
    )


@sh.add_function(dsp, outputs=['ems_data'])
def calculate_ems_data(
        ems_model, times, motive_powers, motors_maximums_powers,
        gear_box_speeds_in, idle_engine_speed):
    """
    Calculate EMS decision data.

    :param ems_model:
        Energy Management Strategy model.
    :type ems_model: EMS

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param motors_maximums_powers:
        Maximum powers of electric motors [kW].
    :type motors_maximums_powers: numpy.array

    :param gear_box_speeds_in:
        Gear box speed [RPM].
    :type gear_box_speeds_in: numpy.array

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :return:
        EMS decision data.
    :rtype: dict
    """
    return ems_model(
        times, motive_powers, motors_maximums_powers, gear_box_speeds_in,
        idle_engine_speed
    )


def _index_anomalies(anomalies):
    i = np.where(np.logical_xor(anomalies[:-1], anomalies[1:]))[0] + 1
    if i.shape[0]:
        if i[0] and anomalies[0]:
            i = np.append([0], i)
        if anomalies[-1]:
            i = np.append(i, [len(anomalies) - 1])
    return i.reshape(-1, 2)


@sh.add_function(dsp, inputs_kwargs=True, outputs=['catalyst_warm_up'])
def identify_catalyst_warm_up(
        times, engine_powers_out, engine_coolant_temperatures, on_engine,
        engine_thermostat_temperature, ems_data, hybrid_modes,
        is_cycle_hot=False):
    """
    Identifies catalyst warm up phase.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param engine_coolant_temperatures:
        Engine coolant temperature vector [°C].
    :type engine_coolant_temperatures: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param ems_data:
        EMS decision data.
    :type ems_data: dict

    :param hybrid_modes:
        Hybrid mode status (0: EV, 1: Parallel, 2: Serial).
    :type hybrid_modes: numpy.array

    :param engine_thermostat_temperature:
        Engine thermostat temperature [°C].
    :type engine_thermostat_temperature: float

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool

    :return:
        Catalyst warm up phase.
    :rtype: numpy.array
    """
    anomalies = np.zeros_like(times)
    if is_cycle_hot or not on_engine.any():
        return anomalies.astype(bool)
    from co2mpas.utils import clear_fluctuations, median_filter
    from sklearn.ensemble import IsolationForest
    i = np.where(on_engine)[0]
    p = np.column_stack((engine_powers_out[i], np.choose(
        ems_data['hybrid_modes'][:, 1] - 1,
        [ems_data[k]['power_ice'].ravel() for k in ('parallel', 'serial')]
    )[i]))

    # noinspection PyUnresolvedReferences
    anomalies[i[IsolationForest(
        random_state=0, behaviour='new', contamination='auto'
    ).fit(p).predict(p) == -1]] = 1
    anomalies = median_filter(times, anomalies, 5)
    anomalies = clear_fluctuations(times, anomalies, 5).astype(bool)
    i, temp = _index_anomalies(anomalies), engine_coolant_temperatures
    b = np.diff(temp[i], axis=1) > 4
    b &= temp[i[:, 0], None] < (engine_thermostat_temperature - 10)
    b &= np.diff(times[i], axis=1) > 5
    b &= np.apply_along_axis(
        lambda a: np.in1d(2, hybrid_modes[slice(*a)]), 1, i
    )
    i = i[b.ravel()]
    anomalies[:] = False
    if i.shape[0]:
        i, j = i[0]
        while i and hybrid_modes.take(i - 1, mode='clip'):
            i -= 1
        anomalies[i:j] = True
    return anomalies


@sh.add_function(dsp, outputs=['catalyst_warm_up_duration'])
def identify_catalyst_warm_up_duration(times, catalyst_warm_up):
    """
    Identify catalyst warm up duration [s].

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param catalyst_warm_up:
        Catalyst warm up phase.
    :type catalyst_warm_up: numpy.array

    :return:
        Catalyst warm up duration [s].
    :rtype: float
    """
    i = _index_anomalies(catalyst_warm_up)
    if i.shape[0]:
        return float(np.diff(times[i[0]]))
    return .0


# noinspection PyMissingOrEmptyDocstring
class StartStopHybrid:
    def __init__(self, params=None):
        self.params = params

    def fit(self, ems_data, on_engine, drive_battery_state_of_charges,
            catalyst_warm_up):
        import lmfit
        k = np.where(~on_engine, *ems_data['k_reference'].T)

        # Filter data.
        b = ~catalyst_warm_up & ~ ems_data['force_on_engine']
        s = np.where(on_engine[b], 1, -1)
        k, soc = k[b].T, drive_battery_state_of_charges[b]
        del b

        def _(x):
            return np.maximum(0, s * (self._k(x.valuesdict(), soc) - k)).sum()

        p = lmfit.Parameters()
        p.add('k0', 0, min=0)
        p.add('soc0', 0, min=0, max=100)
        p.add('alpha', 0, min=0)
        p.add('beta', 0, min=0)
        # noinspection PyUnresolvedReferences
        self.params = lmfit.minimize(_, p, method='ampgo').params.valuesdict()
        return self

    @staticmethod
    def _k(p, soc):
        dsoc = soc - p['soc0']
        return p['alpha'] * dsoc + p['beta'] * dsoc ** 3 + p['k0']

    def __call__(self, drive_battery_state_of_charges):
        return self._k(self.params, drive_battery_state_of_charges)


@sh.add_function(dsp, outputs=['start_stop_hybrid_params'])
def calibrate_start_stop_hybrid_params(
        ems_data, on_engine, drive_battery_state_of_charges, catalyst_warm_up):
    """
    Calibrate start stop model for hybrid electric vehicles.

    :param ems_data:
        EMS decision data.
    :type ems_data: dict

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param drive_battery_state_of_charges:
        State of charge of the drive battery [%].
    :type drive_battery_state_of_charges: numpy.array

    :param catalyst_warm_up:
        Catalyst warm up phase.
    :type catalyst_warm_up: numpy.array

    :return:
        Params of start stop model for hybrid electric vehicles.
    :rtype: dict
    """
    return StartStopHybrid().fit(
        ems_data, on_engine, drive_battery_state_of_charges,
        catalyst_warm_up
    ).params


@sh.add_function(dsp, outputs=['start_stop_hybrid'])
def define_start_stop_hybrid(start_stop_hybrid_params):
    """
    Defines start stop model for hybrid electric vehicles.

    :param start_stop_hybrid_params:
        Params of start stop model for hybrid electric vehicles.
    :type start_stop_hybrid_params: dict

    :return:
        Start stop model for hybrid electric vehicles.
    :rtype: StartStopHybrid
    """
    return StartStopHybrid(start_stop_hybrid_params)


@sh.add_function(dsp, outputs=['hybrid_modes'])
def predict_hybrid_modes(
        start_stop_hybrid, ems_data, drive_battery_model, times, motive_powers,
        accelerations, catalyst_warm_up_duration, start_stop_activation_time,
        min_time_engine_on_after_start, is_cycle_hot):
    """
    Predicts the hybrid mode status (0: EV, 1: Parallel, 2: Serial).

    :param start_stop_hybrid:
        Start stop model for hybrid electric vehicles.
    :type start_stop_hybrid: StartStopHybrid

    :param ems_data:
        EMS decision data.
    :type ems_data: dict

    :param drive_battery_model:
        Drive battery current model.
    :type drive_battery_model: DriveBatteryModel

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param accelerations:
        Acceleration [m/s2].
    :type accelerations: numpy.array

    :param catalyst_warm_up_duration:
        Catalyst warm up duration [s].
    :type catalyst_warm_up_duration: float

    :param min_time_engine_on_after_start:
        Minimum time of engine on after a start [s].
    :type min_time_engine_on_after_start: float

    :param start_stop_activation_time:
        Start-stop activation time threshold [s].
    :type start_stop_activation_time: float

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool

    :return:
        Hybrid mode status (0: EV, 1: Parallel, 2: Serial).
    :rtype: numpy.array
    """
    r = ems_data

    ele, par, ser = r['electric'], r['parallel'], r['serial']
    current_bat = {
        i: v['current_bat'].ravel() for i, v in enumerate((ele, par, ser, ele))
    }
    starter_bat = {
        i: v['power_start'].ravel() for i, v in enumerate((par, ser, ser), 1)
    }
    starter_bat[0] = np.where(
        r['hybrid_modes'].T[0] == 1, *(v['power_stop'].T for v in (par, ser))
    ).ravel()
    from ..electrics.motors.starter import calculate_starter_currents as func
    nom_volt = drive_battery_model.service.nominal_voltage
    starter_bat = {k: func(v, nom_volt) for k, v in starter_bat.items()}

    # noinspection PyUnresolvedReferences
    hybrid_modes = r['force_on_engine'].astype(int)
    drive_battery_model.reset()
    soc, t0 = drive_battery_model.init_soc, start_stop_activation_time
    it = enumerate(zip(
        times, motive_powers, accelerations, hybrid_modes, r['k_reference'],
        r['hybrid_modes']
    ))
    is_warm = is_cycle_hot
    for i, (t, motive_power, acc, mode, k_ref, mode_ref) in it:
        pre_mode = hybrid_modes.take(i - 1, mode='clip')
        j = int(bool(pre_mode))
        if not mode and (t < t0 or k_ref[j] > start_stop_hybrid(soc)):
            mode = mode_ref[j]
        starter_curr = 0
        if bool(pre_mode) ^ bool(mode) and i:
            if mode:
                t0 = t + min_time_engine_on_after_start
                if not is_warm:
                    if not hybrid_modes[i]:
                        mode = 3
                    j = np.searchsorted(times, t + catalyst_warm_up_duration)
                    j += 1
                    hybrid_modes[i:j][hybrid_modes[i:j] == 0], is_warm = 3, True
            starter_curr = starter_bat[mode][i]
        soc = drive_battery_model(
            current_bat[mode][i], t, motive_power, acc, bool(mode), starter_curr
        )
        hybrid_modes[i] = mode
    return np.minimum(hybrid_modes, 2)


@sh.add_function(dsp, outputs=['on_engine'])
def identify_on_engine(hybrid_modes):
    """
    Identifies if the engine is on [-].

    :param hybrid_modes:
        Hybrid mode status (0: EV, 1: Parallel, 2: Serial).
    :type hybrid_modes: numpy.array

    :return:
        If the engine is on [-].
    :rtype: numpy.array
    """
    return hybrid_modes.astype(bool)


@sh.add_function(dsp, inputs_kwargs=True, outputs=['catalyst_warm_up'])
def predict_catalyst_warm_up(
        times, on_engine, catalyst_warm_up_duration, is_cycle_hot=False):
    """
    Predict catalyst warm up phase.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param on_engine:
        If the engine is on [-].
    :type on_engine: numpy.array

    :param catalyst_warm_up_duration:
        Catalyst warm up duration [s].
    :type catalyst_warm_up_duration: float

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool, optional

    :return:
        Catalyst warm up phase.
    :rtype: numpy.array
    """
    if not is_cycle_hot and on_engine.any():
        t1 = times[on_engine][0] + catalyst_warm_up_duration
        return on_engine & (times <= t1)
    return np.zeros_like(times, bool)


@sh.add_function(dsp, outputs=['catalyst_warm_up'])
def predict_catalyst_warm_up_v1(times, is_cycle_hot):
    """
    Predict catalyst warm up phase.

    :param times:
        Time vector [s].
    :type times: numpy.array

    :param is_cycle_hot:
        Is an hot cycle?
    :type is_cycle_hot: bool, optional

    :return:
        Catalyst warm up phase.
    :rtype: numpy.array
    """
    if is_cycle_hot:
        return np.zeros_like(times, bool)
    return sh.NONE


@sh.add_function(dsp, outputs=['engine_speeds_out_hot'], weight=2)
def predict_engine_speeds_out_hot(ems_data, hybrid_modes):
    """
    Predicts the engine speed at hot condition [RPM].

    :param ems_data:
        EMS decision data.
    :type ems_data: dict

    :param hybrid_modes:
        Hybrid mode status (0: EV, 1: Parallel, 2: Serial).
    :type hybrid_modes: numpy.array

    :return:
        Engine speed at hot condition [RPM].
    :rtype: numpy.array
    """
    it = ems_data['electric'], ems_data['parallel'], ems_data['serial']
    return np.choose(hybrid_modes, [d['speed_ice'].ravel() for d in it])


@sh.add_function(dsp, outputs=[
    'motor_p4_electric_powers', 'motor_p3_electric_powers',
    'motor_p2_electric_powers', 'motor_p1_electric_powers',
    'motor_p0_electric_powers'
])
def predict_motors_electric_powers(
        ems_data, catalyst_warm_up, hybrid_modes, engine_speeds_out_hot):
    """
    Predicts motors electric power split [kW].

    :param ems_data:
        EMS decision data.
    :type ems_data: dict

    :param catalyst_warm_up:
        Catalyst warm up phase.
    :type catalyst_warm_up: numpy.array

    :param hybrid_modes:
        Hybrid mode status (0: EV, 1: Parallel, 2: Serial).
    :type hybrid_modes: numpy.array

    :param engine_speeds_out_hot:
        Engine speed at hot condition [RPM].
    :type engine_speeds_out_hot: numpy.array

    :return:
        Motors electric powers [kW].
    :rtype: tuple[numpy.array]
    """
    mode = np.where(catalyst_warm_up & (hybrid_modes == 2), 0, hybrid_modes)
    return np.choose(mode, [
        d['battery_power_split'](d['power_bat'].ravel(), engine_speeds_out_hot)
        for d in (ems_data[k] for k in ('electric', 'parallel', 'serial'))
    ])
