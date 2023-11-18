import dmlab2d
from gymnasium import spaces
import numpy as np
from ray.rllib.env import multi_agent_env
import json
from  matplotlib import pyplot as plt
from functools import cache

from baselines.train import utils

PLAYER_STR_FORMAT = 'player_{index}'


class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
  """Interfacing Melting Pot substrates and RLLib MultiAgentEnv."""

  def __init__(self, env: dmlab2d.Environment):
    """Initializes the instance.

    Args:
      env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
    """
    self.debug_out = []
    self.rgb_map = {}
    self.img_count = 0
    self._env = env
    self._num_players = len(self._env.observation_spec())
    self._ordered_agent_ids = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    # RLLib requires environments to have the following member variables:
    # observation_space, action_space, and _agent_ids
    self._agent_ids = set(self._ordered_agent_ids)
    
    # RLLib expects a dictionary of agent_id to observation or action,
    # Melting Pot uses a tuple, so we convert them here
    self.observation_space = self._convert_spaces_tuple_to_dict(
        utils.spec_to_space(self._env.observation_spec()),
        remove_world_observations=True)
    self.action_space = self._convert_spaces_tuple_to_dict(
        utils.spec_to_space(self._env.action_spec()))
    super().__init__()

  def reset(self, *args, **kwargs):
    """See base class."""
    timestep = self._env.reset()
    return utils.timestep_to_observations(timestep), {}

  def step(self, action_dict):
    """See base class."""
    NOOP = 0
    FORWARD = 1
    BACKWARD = 2
    STEP_LEFT = 3
    STEP_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    FIRE_ZAP = 7
    FIRE_CLEAN = 8

    action_set = {
        0: "NOOP",
        1: "FORWARD",
        2: "BACKWARD",
        3: "STEP_LEFT",
        4: "STEP_RIGHT",
        5: "TURN_LEFT",
        6: "TURN_RIGHT",
        7: "FIRE_ZAP",
        8: "FIRE_CLEAN"
    }

    DIRTY_WATER = [28, 152, 147]
    CLEANING_AREA = [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    # @cache
    def generateCoordsFromBitmap(bitmap):
      coords = []
      for row in range(len(bitmap)):
        for col in range(len(bitmap[0])):
          if bitmap[row][col] == 1:
            coords.append((row, col))
      return coords

    def isDirtyWaterInRange(rgb):
      cleaning_coords = generateCoordsFromBitmap(CLEANING_AREA)
      for row, col in cleaning_coords:
        if rgb[row][col].tolist() == DIRTY_WATER:
          return True
      return False

    def rewardFunc(action, observation):
      rgb = observation["RGB"]
      should_clean = isDirtyWaterInRange(rgb)
      if should_clean:
        if action == FIRE_CLEAN:
          return 1
        else:
          return -1
      else:
        return -0.25

    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    observations = utils.timestep_to_observations(timestep)

    # player_0_rgb = observations["player_0"]["RGB"]
    # should_player_0_clean = isDirtyWaterInRange(player_0_rgb)
    # if should_player_0_clean:
    #   print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    #   print("CLEAN!!!!!")
    #   print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    #   plt.imshow(player_0_rgb)
    #   plt.show()

    p0_r = rewardFunc(action_dict["player_0"], observations["player_0"])
    plt.imshow(observations["player_0"]["RGB"])
    plt.savefig(f"./img_out/{self.img_count}_{p0_r}.png")
    self.img_count += 1 
    rewards = {
        agent_id: rewardFunc(action_dict[agent_id], observations[agent_id])
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }
    done = {'__all__': timestep.last()}
    info = {}

    # for val in observations.values():
    #   rgb = json.dumps(val['RGB'].tolist())
    #   if rgb not in self.rgb_map:
    #     self.rgb_map[rgb] = len(self.rgb_map)

    debug_rewards = {
      agent_id: reward
      for agent_id, reward in rewards.items()
    }
    debug_observations = {
      agent_id: {
        'COLLECTIVE_REWARD': val['COLLECTIVE_REWARD'],
        'READY_TO_SHOOT': val['READY_TO_SHOOT'].tolist(),
        # 'RGB': self.rgb_map[json.dumps(val['RGB'].tolist())]
      }
      for agent_id, val in observations.items()
    }
    debug = {
      "actions": list(map(lambda x: action_set[x], actions)),
      "rewards": debug_rewards,
      "observations": debug_observations
    }
    self.debug_out.append(debug)
    
    return observations, rewards, done, done, info

  def close(self):
    """See base class."""
    print("debug length:", len(self.debug_out))
    if len(self.debug_out) > 0:
      with open("./my_debug_out.json", "w") as outfile:
          json_object = json.dumps(self.debug_out, indent=5)
          outfile.write(json_object)

    with open("./rgb_mapping.json", "w") as outfile:
        json_object = json.dumps(self.rgb_map, indent=5)
        outfile.write(json_object)
    self._env.close()

  def get_dmlab2d_env(self):
    """Returns the underlying DM Lab2D environment."""

    return self._env

  # Metadata is required by the gym `Env` class that we are extending, to show
  # which modes the `render` method supports.
  metadata = {'render.modes': ['rgb_array']}

  def render(self) -> np.ndarray:
    """Render the environment.

    This allows you to set `record_env` in your training config, to record
    videos of gameplay.

    Returns:
        np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable for turning
        into a video.
    """

    observation = self._env.observation()
    world_rgb = observation[0]['WORLD.RGB']

    # RGB mode is used for recording videos
    return world_rgb

  def _convert_spaces_tuple_to_dict(
      self,
      input_tuple: spaces.Tuple,
      remove_world_observations: bool = False) -> spaces.Dict:
    """Returns spaces tuple converted to a dictionary.

    Args:
      input_tuple: tuple to convert.
      remove_world_observations: If True will remove non-player observations.
    """

    return spaces.Dict({
        agent_id: (utils.remove_unrequired_observations_from_space(input_tuple[i])
                   if remove_world_observations else input_tuple[i])
        for i, agent_id in enumerate(self._ordered_agent_ids)
    })
