from gym import spaces
import numpy as np

DISCRETE_SPACES = (
    spaces.Discrete,
    spaces.MultiBinary,
    spaces.MultiDiscrete,
)
CONTINUOUS_SPACES = (spaces.Box, )


def is_continuous_space(space):
    return isinstance(space, CONTINUOUS_SPACES)


def is_discrete_space(space):
    return isinstance(space, DISCRETE_SPACES)


def heuristic_target_entropy(action_space):
    if is_continuous_space(action_space):
        heuristic_target_entropy = -np.prod(action_space.shape)
    elif is_discrete_space(action_space):
        raise NotImplementedError(
            "TODO(hartikainen): implement for discrete spaces.")
    else:
        raise NotImplementedError((type(action_space), action_space))

    return heuristic_target_entropy
