# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and `dsp` model to model the GSPV Approach.
"""
import copy
import numpy as np
import schedula as sh
from .cmv import CMV
import co2mpas.utils as co2_utl
from co2mpas.defaults import dfl
from .core import prediction_gears_gsm

dsp = sh.BlueDispatcher(name='Gear Shifting Power Velocity Approach')
dsp.add_data('stop_velocity', dfl.values.stop_velocity)


def _gspv_interpolate_cloud(powers, velocities):
    from sklearn.isotonic import IsotonicRegression
    from scipy.interpolate import InterpolatedUnivariateSpline
    regressor = IsotonicRegression()
    regressor.fit(powers, velocities)
    x = np.linspace(min(powers), max(powers))
    y = regressor.predict(x)
    return InterpolatedUnivariateSpline(x, y, k=1, ext=3)


# noinspection PyMissingOrEmptyDocstring,PyPep8Naming
class GSPV(CMV):
    def __init__(self, *args, cloud=None, velocity_speed_ratios=None):
        super(GSPV, self).__init__(*args)
        if args and isinstance(args[0], GSPV):
            if not cloud:
                self.cloud = args[0].cloud
            if velocity_speed_ratios:
                self.convert(velocity_speed_ratios)
            else:
                velocity_speed_ratios = args[0].velocity_speed_ratios
        else:
            self.cloud = cloud or {}

        self.velocity_speed_ratios = velocity_speed_ratios or {}
        if cloud:
            self._fit_cloud()

    def __repr__(self):
        from pprint import pformat
        s = 'GSPV(cloud={}, velocity_speed_ratios={})'
        vsr = pformat(self.velocity_speed_ratios)
        s = s.format(pformat(self.cloud), vsr)
        return s.replace('inf', "float('inf')")

    # noinspection PyMethodOverriding
    def fit(self, gears, velocities, motive_powers, velocity_speed_ratios,
            stop_velocity):
        self.clear()

        self.velocity_speed_ratios = velocity_speed_ratios

        it = zip(velocities, motive_powers, co2_utl.pairwise(gears))

        for v, p, (g0, g1) in it:
            if v > stop_velocity and g0 != g1:
                x = self.get(g0, [[], [[], []]])
                if g0 < g1 and p >= 0:
                    x[1][0].append(p)
                    x[1][1].append(v)
                elif g0 > g1 and p <= 0:
                    x[0].append(v)
                else:
                    continue
                self[g0] = x

        self[0] = [[0.0], [[0.0], [stop_velocity]]]

        self[max(self)][1] = [[0, 1], [dfl.INF] * 2]

        self.cloud = {k: copy.deepcopy(v) for k, v in self.items()}

        self._fit_cloud()

        return self

    def _fit_cloud(self):
        from scipy.interpolate import InterpolatedUnivariateSpline as spl

        def _line(n, m, i):
            x = np.mean(m[i]) if m[i] else None
            k_p = n - 1
            while k_p > 0 and k_p not in self:
                k_p -= 1
            x_up = self[k_p][not i](0) if k_p >= 0 else x

            if x is None or x > x_up:
                x = x_up
            return spl([0, 1], [x] * 2, k=1)

        self.clear()
        self.update(copy.deepcopy(self.cloud))

        for k, v in sorted(self.items()):
            v[0] = _line(k, v, 0)

            if len(v[1][0]) > 2:
                v[1] = _gspv_interpolate_cloud(*v[1])
            elif v[1][1]:
                v[1] = spl([0, 1], [np.mean(v[1][1])] * 2, k=1)
            else:
                v[1] = self[k - 1][0]

    @property
    def limits(self):
        limits = {}
        X = [dfl.INF, 0]
        for v in self.cloud.values():
            X[0] = min(min(v[1][0]), X[0])
            X[1] = max(max(v[1][0]), X[1])
        X = list(np.linspace(*X))
        X = [0] + X + [X[-1] * 1.1]
        for k, func in self.items():
            limits[k] = [(f(X), X) for f, x in zip(func, X)]
        return limits

    def fig(self):
        import itertools
        import plotly.graph_objs as go
        from plotly.colors import DEFAULT_PLOTLY_COLORS
        colors = itertools.cycle(DEFAULT_PLOTLY_COLORS)
        fig = go.Figure()
        for k, v in self.limits.items():
            color = next(colors)
            for (s, l), (x, y) in zip((('down', 'dash'), ('up', None)), v):
                if x[0] < dfl.INF:
                    fig.add_trace(go.Scatter(
                        name='Gear %d:%s-shift' % (k, s), x=x, y=y,
                        line=dict(color=color, dash=l), mode='lines'
                    ))
            cy, cx = self.cloud[k][1]
            if cx[0] < dfl.INF:
                fig.add_trace(go.Scatter(
                    x=cx, y=cy, marker=dict(color=color), mode='markers',
                    showlegend=False
                ))
        fig.update_layout(
            title=self.__class__.__name__,
            xaxis_title="Velocity [km/h]",
            yaxis_title="Power [kW]"
        )
        return fig

    def _init_gear(self, times, velocities, accelerations, motive_powers,
                   engine_coolant_temperatures):
        keys = sorted(self.keys())
        matrix, c = {}, len(keys) - 1
        from co2mpas.utils import List
        if isinstance(velocities, List) or isinstance(motive_powers, List):
            for i, k in enumerate(keys):
                matrix[k] = self[k], (keys[max(0, i - 1)], keys[min(i + 1, c)])

            def _next(gear, index):
                # noinspection PyShadowingNames
                v, p = velocities[index], motive_powers[index]
                (_down, _up), (g0, g1) = matrix[gear]
                if v >= _up(p):
                    return g1
                if v < _down(p):
                    return g0
                return gear
        else:
            r = velocities.shape[0]
            for i, g in enumerate(keys):
                down, up = [func(motive_powers) for func in self[g]]
                matrix[g] = p = np.tile(g, r)
                p[velocities < down] = keys[max(0, i - 1)]
                p[velocities >= up] = keys[min(i + 1, c)]

            def _next(gear, index):
                return matrix[gear][index]
        return _next

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def convert(self, velocity_speed_ratios):
        if velocity_speed_ratios != self.velocity_speed_ratios:
            # noinspection PyProtectedMember
            from .cmv import _convert_limits
            vsr, n_vsr = self.velocity_speed_ratios, velocity_speed_ratios

            limits = [dfl.INF, 0]

            for v in self.cloud.values():
                if v[1][0]:
                    limits[0] = min(min(v[1][0]), limits[0])
                    limits[1] = max(max(v[1][0]), limits[1])

            cloud = self.cloud = {}

            for p in np.linspace(*limits):
                cmv = _convert_limits(
                    {k: [f(p) for f in v] for k, v in self.items()}, vsr, n_vsr
                )

                for k, (l, u) in sorted(cmv.items()):
                    c = cloud[k] = cloud.get(k, [[], [[], []]])
                    c[0].append(l)
                    c[1][0].append(p)
                    c[1][1].append(u)

            cloud[0] = [[0.0], [[0.0], [self[0][1](0.0)]]]
            cloud[max(cloud)][1] = [[0, 1], [dfl.INF] * 2]

            self._fit_cloud()

            self.velocity_speed_ratios = n_vsr

        return self


@sh.add_function(dsp, outputs=['GSPV'])
def calibrate_gspv(
        gears, velocities, motive_powers, velocity_speed_ratios, stop_velocity):
    """
    Identifies gear shifting power velocity matrix.

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array

    :param motive_powers:
        Motive power [kW].
    :type motive_powers: numpy.array

    :param velocity_speed_ratios:
        Constant velocity speed ratios of the gear box [km/(h*RPM)].
    :type velocity_speed_ratios: dict[int | float]

    :param stop_velocity:
        Maximum velocity to consider the vehicle stopped [km/h].
    :type stop_velocity: float

    :return:
        Gear shifting power velocity matrix.
    :rtype: dict
    """

    gspv = GSPV().fit(
        gears, velocities, motive_powers, velocity_speed_ratios, stop_velocity
    )

    return gspv


# predict gears with corrected matrix velocity
dsp.add_function(
    function=prediction_gears_gsm,
    inputs=[
        'correct_gear', 'gear_filter', 'GSPV', 'times', 'velocities',
        'accelerations', 'motive_powers', 'cycle_type', 'velocity_speed_ratios'
    ],
    outputs=['gears']
)
