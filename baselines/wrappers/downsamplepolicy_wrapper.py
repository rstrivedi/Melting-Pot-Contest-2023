import dm_env
from baselines.train import utils
from meltingpot.utils.policies import policy
from typing import Any

def _downsample_single_timestep(timestep: dm_env.TimeStep, scale) -> dm_env.TimeStep:
    return timestep._replace(
        observation={k: utils.downsample_observation(v, scale) if k == 'RGB' else v for k, v in timestep.observation.items()
        })

class DownsamplingPolicyWraper(policy.Policy[policy.State]):
  """Downsample observation before providing as input to agent policies during evaluation.
  
  This is required during evaluation when the policy is trained on Downsampled observations using
  the DownSamplingSubstrateWrapper. That is because the scenarios does not downsample observations
  and hence during evaluation focal population policies will receive full size oobservation which
  they won't be able to handle without this wrapper.
  """

  def __init__(self, policy: policy.Policy, scale):
    self._policy = policy
    self._scale = scale

  def step(self, timestep: dm_env.TimeStep, prev_state: policy.State) -> Any:
    return self._policy.step(_downsample_single_timestep(timestep, self._scale), prev_state)

  def initial_state(self) -> Any:
    return self._policy.initial_state()

  def close(self) -> None:
    self._policy.close()