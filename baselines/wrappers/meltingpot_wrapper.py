import dmlab2d
from gymnasium import spaces
import numpy as np
from ray.rllib.env import multi_agent_env
import json
from  matplotlib import pyplot as plt
from functools import cache
from collections import deque
import math
import sys

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

    # CONSTANTS
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

    # RGB values for sprites of interest
    DIRTY_WATER = [28, 152, 147]
    APPLE = [171, 153, 69]

    # Bit map representing area of effect for FIRE_CLEAN action
    CLEANING_AREA = (
      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
      (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
      (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    
    @cache
    def generateCoordsFromBitmap(bitmap):
      """
      Memoized function to convert bitmap to a list of coordinates.
      """
      coords = []
      for row in range(len(bitmap)):
        for col in range(len(bitmap[0])):
          if bitmap[row][col] == 1:
            coords.append((row, col))
      return coords

    def isDirtyWaterInRange(rgb):
      """
      Check if there exists at least one dirty water cell in the FIRE_CLEAN area of effect.
      """
      cleaning_coords = generateCoordsFromBitmap(CLEANING_AREA)
      for row, col in cleaning_coords:
        if rgb[row][col].tolist() == DIRTY_WATER:
          return True
      return False

    def distanceToNearestApple(rgb):
      """
      Computes Manhattan distance to the nearest apple cell from the player.
      """
      directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
      queue = deque()
      queue.append((9, 5))
      visited = set()
      while queue:
        cur = queue.popleft()
        if cur not in visited:
          visited.add(cur)
          row, col = cur
          # print(row, col)
          if rgb[row][col].tolist() == APPLE:
            dist_to_nearest_apple = abs(col - 5) + abs(row - 9)

            ## DEBUG
            # dist = "{:.6f}".format(dist_to_nearest_apple)
            # mapp = rgb
            # for v in visited:
            #   mapp[v[0]][v[1]][0] += 20
            #   mapp[v[0]][v[1]][1] += 20
            #   mapp[v[0]][v[1]][2] += 20
            # plt.imshow(mapp)
            # plt.savefig(f"./img_out/{self.img_count}_{row}_{col}_{dist}.png")
            # self.img_count += 1 

            return dist_to_nearest_apple
          
          for d_row, d_col in directions:
            new_row = row + d_row
            new_col = col + d_col
            if new_row >= 0 and new_row < 11 and new_col >= 0 and new_col < 11:
              queue.append((row + d_row, col + d_col))
      return -1
          
    def rewardFunc(action, observation):
      """
      Custom reward function.
      If an apple is in FOV, return 1 / distance_to_apple.
      If dirty water is in the cleaning area of effect, then reward the agent
      for doing FIRE_CLEAN, otherwise punish them for doing anything else.
      """
      rgb = observation["RGB"]
      dist_to_nearest_apple = distanceToNearestApple(rgb)
      should_clean = isDirtyWaterInRange(rgb)
      if dist_to_nearest_apple != -1:
        return 1 / dist_to_nearest_apple

      if should_clean:
        if action == FIRE_CLEAN:
          return 1
        else:
          return -1
      else:
        return -0.1

    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    observations = utils.timestep_to_observations(timestep)

    # # DEBUG: Visualize what player_0 is doing
    # p0_r = rewardFunc(action_dict["player_0"], observations["player_0"])
    # p0_r = "{:.6f}".format(p0_r)
    # plt.imshow(observations["player_0"]["RGB"])
    # plt.savefig(f"./img_out/{self.img_count}_{p0_r}.png")
    # self.img_count += 1
  
    rewards = {
        agent_id: rewardFunc(action_dict[agent_id], observations[agent_id])
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }
    done = {'__all__': timestep.last()}
    info = {}

    ###### START DEBUG LOGS ######
    debug_rewards = {
      agent_id: reward
      for agent_id, reward in rewards.items()
    }
    debug_observations = {
      agent_id: {
        'COLLECTIVE_REWARD': val['COLLECTIVE_REWARD'],
        'READY_TO_SHOOT': val['READY_TO_SHOOT'].tolist(),
      }
      for agent_id, val in observations.items()
    }
    debug = {
      "actions": list(map(lambda x: action_set[x], actions)),
      "rewards": debug_rewards,
      "observations": debug_observations
    }
    self.debug_out.append(debug)
    ###### END DEBUG LOGS ######
    
    return observations, rewards, done, done, info

  def close(self):
    """See base class."""
    # DEBUG: Log actions and observations at each time step to JSON file
    # print("debug length:", len(self.debug_out))
    # if len(self.debug_out) > 0:
    #   with open("./my_debug_out.json", "w") as outfile:
    #       json_object = json.dumps(self.debug_out, indent=5)
    #       outfile.write(json_object)
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
