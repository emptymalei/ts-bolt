import math

from ts_bolt.physics.pendulum import Pendulum


def test_pendulum_generate_data():  # type: ignore
    pendulum = Pendulum(length=1.0, gravity=9.81)

    assert pendulum.length == 1.0
    assert pendulum.gravity == 9.81
    assert pendulum.period == 2 * math.pi * math.sqrt(
        pendulum.length / pendulum.gravity
    )

    num_periods = 2
    num_samples_per_period = 10
    data = pendulum(num_periods, num_samples_per_period)

    assert len(data["t"]) == num_periods * num_samples_per_period
