import numpy as np
import dm_env

from baselines.train import utils
from meltingpot.utils.substrates.wrappers import observables
from meltingpot.utils.substrates import substrate
from collections.abc import Mapping, Sequence
from typing import Any

def _downsample_multi_timestep(timestep: dm_env.TimeStep, scaled) -> dm_env.TimeStep:
    return timestep._replace(
        observation=[{k: utils.downsample_observation(v, scaled) if k == 'RGB' else v for k, v in observation.items()
        } for observation in timestep.observation])

def _downsample_multi_spec(spec, scaled):
    return dm_env.specs.Array(shape=(spec.shape[0]//scaled, spec.shape[1]//scaled, spec.shape[2]), dtype=spec.dtype)

class DownSamplingSubstrateWrapper(observables.ObservableLab2dWrapper):
    """Downsamples 8x8 sprites returned by substrate to 1x1. 
    
    This related to the observation window of each agent and will lead to observation RGB shape to reduce
    from [88, 88, 3] to [11, 11, 3]. Other downsampling scales are allowed but not tested. Thsi will lead
    to significant speedups in training.
    """

    def __init__(self, substrate: substrate.Substrate, scaled):
        super().__init__(substrate)
        self._scaled = scaled

    def reset(self) -> dm_env.TimeStep:
        timestep = super().reset()
        return _downsample_multi_timestep(timestep, self._scaled)

    def step(self, actions) -> dm_env.TimeStep:
        timestep = super().step(actions)
        return _downsample_multi_timestep(timestep, self._scaled)

    def observation_spec(self) -> Sequence[Mapping[str, Any]]:
        spec = super().observation_spec()
        return [{k: _downsample_multi_spec(v, self._scaled) if k == 'RGB' else v for k, v in s.items()}
        for s in spec]