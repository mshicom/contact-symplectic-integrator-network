#%%
import sys

sys.path.append('../')
from environments import Pendulum, BouncingBall, NewtonCradle


def test_pendulum_shapes():
    test_steps = 400
    test_horizon = 10
    env = Pendulum(steps=test_steps, horizon=test_horizon, dt=0.01, epochs=10, friction=0., length=1, SIGMA=0.1)
    env.generate()
    assert env.trajectory.shape[0] == test_steps
    assert env.trajectory.shape[1] == 3
    assert env.y.shape[0] == test_steps - test_horizon - 1
    assert env.X.shape[0] == test_steps - test_horizon - 1
    assert env.X.shape[1] == 2
    assert env.y.shape[1] == 2 * test_horizon

    test_horizon = 5
    env = Pendulum(steps=test_steps, horizon=test_horizon, dt=0.01, epochs=10, friction=0., length=1, SIGMA=0.1)
    env.generate()
    assert env.y.shape[0] == test_steps - test_horizon - 1
    assert env.y.shape[1] == 2 * test_horizon


def test_bouncing_ball_shapes():
    test_steps = 400
    test_horizon = 10
    env = BouncingBall(steps=test_steps - 1, horizon=test_horizon, dt=0.01, epochs=10, SIGMA=0.1)
    env.generate()
    assert env.trajectory.shape[0] == test_steps
    assert env.trajectory.shape[1] == 3
    assert env.y.shape[0] == test_steps - test_horizon - 1
    assert env.X.shape[0] == test_steps - test_horizon - 1
    assert env.X.shape[1] == 2
    assert env.y.shape[1] == 2 * test_horizon

    test_horizon = 5
    env = BouncingBall(steps=test_steps - 1, horizon=test_horizon, dt=0.01, epochs=10, SIGMA=0.1)
    env.generate()
    assert env.y.shape[0] == test_steps - test_horizon - 1
    assert env.y.shape[1] == 2 * test_horizon


def test_newton_cradle_shapes():
    test_steps = 400
    test_horizon = 10
    env = NewtonCradle(steps=test_steps - 1, horizon=test_horizon, dt=0.01, epochs=10, SIGMA=0.1)
    env.generate()
    assert env.trajectory.shape[0] == test_steps
    assert env.trajectory.shape[1] == 5
    assert env.y.shape[0] == test_steps - test_horizon - 1
    assert env.X.shape[0] == test_steps - test_horizon - 1
    assert env.X.shape[1] == 4
    assert env.y.shape[1] == 4 * test_horizon

    test_horizon = 5
    env = NewtonCradle(steps=test_steps - 1, horizon=test_horizon, dt=0.01, epochs=10, SIGMA=0.1)
    env.generate()
    assert env.y.shape[0] == test_steps - test_horizon - 1
    assert env.y.shape[1] == 4 * test_horizon
