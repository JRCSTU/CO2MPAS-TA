import numpy as np
import schedula as sh
from ..defaults import dfl

dsp = sh.BlueDispatcher(name='Driver logic')


@sh.add_function(dsp, outputs=['previous_velocity', 'previous_time'])
def get_previous(simulation_model):
    return simulation_model.select('velocities', 'times', di=-1)


@sh.add_function(dsp, outputs=['distance', 'velocity', 'time', 'angle_slope'])
def get_current(simulation_model):
    return simulation_model.select(
        'distances', 'velocities', 'times', 'angle_slopes'
    )


@sh.add_function(dsp, outputs=['desired_velocity', 'maximum_distance'])
def calculate_desired_velocity(path_distances, path_velocities, distance):
    """
    Returns the desired velocity [km/h].

    :param path_distances:
        Cumulative distance vector [m].
    :type path_distances: numpy.array

    :param desired_velocities:
        Desired velocity vector [km/h].
    :type desired_velocities: numpy.array

    :param distance:
        Current travelled distance [m].
    :type distance: float

    :return:
        Desired velocity [km/h].
    :rtype: float
    """
    i = np.searchsorted(path_distances, distance, side='right')
    d = path_distances.take(i + 1, mode='clip') + dfl.EPS
    return path_velocities.take(i, mode='clip'), d


@sh.add_function(dsp, outputs=['maximum_power'])
def calculate_maximum_power(
        simulation_model, time, full_load_curve, time_sample_frequency):
    simulation_model(1, time + time_sample_frequency)
    return full_load_curve(simulation_model.select('engine_speeds_out_hot'))


@sh.add_function(dsp, outputs=['max_acceleration_model'])
def define_max_acceleration_model(
        road_loads, vehicle_mass, inertial_factor, static_friction,
        wheel_drive_load_fraction):
    from ..vehicle import _compile_traction_acceleration_limits
    from numpy.polynomial.polynomial import polyroots
    f0, f1, f2 = road_loads
    _m = vehicle_mass * (1 + inertial_factor / 100)
    _b = vehicle_mass * 9.81
    acc_lim = _compile_traction_acceleration_limits(
        static_friction, wheel_drive_load_fraction
    )
    def _func(previous_velocity, next_time, previous_time, angle_slope,
              motive_power):
        dt = (next_time - previous_time) * 3.6
        m = _m / dt

        b = f0 * np.cos(angle_slope) + _b * np.sin(angle_slope)
        b -= m * previous_velocity

        vel = max(polyroots((-motive_power * 3600, b, f1 + m, f2)))
        return np.clip((vel - previous_velocity) / dt, *acc_lim(angle_slope))

    return _func


@sh.add_function(dsp, outputs=['maximum_acceleration'])
def calculate_maximum_acceleration(
        simulation_model, maximum_power, max_acceleration_model, angle_slope,
        previous_velocity, previous_time):
    acc = max_acceleration_model(
        previous_velocity, simulation_model.select('times', di=1),
        previous_time, angle_slope, maximum_power
    )
    return acc


@sh.add_function(dsp, outputs=['acceleration_damping'])
def calculate_acceleration_damping(previous_velocity, desired_velocity):
    """
    Calculates the acceleration damping [-].

    :param velocity:
        Current velocity [km/h].
    :type velocity: float

    :param desired_velocity:
        Desired velocity [km/h].
    :type desired_velocity: float

    :return:
        Acceleration damping factor [-].
    :rtype: float
    """
    r = previous_velocity / desired_velocity
    if r >= 1:
        return 10 * (1 - r)  # Deceleration.
    if r > 0.5:
        return 1 - np.power(r, 60)  # Acceleration.
    return 1 - 0.8 * np.power(1 - r, 60)  # Acceleration boost.


@sh.add_function(dsp, outputs=['acceleration'])
def calculate_desired_acceleration(
        maximum_acceleration, driver_style_ratio, acceleration_damping):
    """
    Calculate the desired acceleration [m/s].

    :param maximum_accelerations:
        Maximum achievable acceleration [m/s].
    :type maximum_accelerations: float | numpy.array

    :param driver_style_ratio:
        Driver style ratio [-].
    :type driver_style_ratio: float

    :param accelerations_damping:
        Acceleration damping factor [-].
    :type accelerations_damping: float | numpy.array

    :return:
        Desired acceleration [m/s].
    :rtype: float
    """
    return maximum_acceleration * driver_style_ratio * acceleration_damping


@sh.add_function(dsp, outputs=['next_time'])
def calculate_next_time(
        time_sample_frequency, time, previous_time, velocity, acceleration,
        previous_velocity, maximum_distance, distance):
    from numpy.polynomial.polynomial import polyroots
    v, a = (velocity + previous_velocity) / 3.6, acceleration
    p = 2 * (distance - maximum_distance), a * (time - previous_time) + v, a
    return time + min(max(polyroots(p)), time_sample_frequency)
