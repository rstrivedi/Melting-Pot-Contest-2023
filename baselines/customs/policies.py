from typing import Tuple

import dm_env
from meltingpot.utils.policies import policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy import sample_batch

_IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']

class EvalPolicy(policy.Policy):
  """ Loads the policies from  Policy checkpoints and removes unrequired observations
  that policies cannot expect to have access to during evaluation.
  """
  def __init__(self,
               chkpt_dir: str,
               policy_id: str = sample_batch.DEFAULT_POLICY_ID) -> None:
    
    policy_path = f'{chkpt_dir}/{policy_id}'
    self._policy = Policy.from_checkpoint(policy_path)
    self._prev_action = 0
  
  def initial_state(self) -> policy.State:
    """See base class."""

    self._prev_action = 0
    state = self._policy.get_initial_state()
    self.prev_state = state
    return state

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""

    observations = {
        key: value
        for key, value in timestep.observation.items()
        if key not in _IGNORE_KEYS
    }

    # We want the logic to be stateless so don't use prev_state from input
    action, state, _ = self._policy.compute_single_action(
        observations,
        self.prev_state,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)

    self._prev_action = action
    self.prev_state = state
    return action, state

  def close(self) -> None:

    """See base class."""