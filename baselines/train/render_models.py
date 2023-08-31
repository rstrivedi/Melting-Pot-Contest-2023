import argparse
import json
import ray
import dm_env
from baselines.customs.policies import EvalPolicy
from dmlab2d.ui_renderer import pygame
import numpy as np
from ray.tune.registry import register_env
import make_envs


def render_model(args):
  
  ray.init()
  register_env("meltingpot", make_envs.env_creator)
  config_file = f'{args.config_dir}/params.json'
  f = open(config_file)
  configs = json.load(f)

  env = make_envs.env_creator(configs["env_config"]).get_dmlab2d_env()

  # Instantiate policies for bots from stored policy checkpoints
  bots = [
      EvalPolicy(args.policies_dir, f"agent_{i}")
      for i in range(len(configs["env_config"]["roles"]))
  ]

  timestep = env.reset()
  states = [bot.initial_state() for bot in bots]
  actions = [0] * len(bots)

  # Configure the pygame display
  scale = 4
  fps = 5

  pygame.init()
  clock = pygame.time.Clock()
  pygame.display.set_caption("DM Lab2d")
  obs_spec = env.observation_spec()
  shape = obs_spec[0]["WORLD.RGB"].shape
  game_display = pygame.display.set_mode(
      (int(shape[1] * scale), int(shape[0] * scale)))

  for k in range(args.horizon):
    obs = timestep.observation[0]["WORLD.RGB"]
    obs = np.transpose(obs, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    rect = surface.get_rect()
    surf = pygame.transform.scale(surface,
                                  (int(rect[2] * scale), int(rect[3] * scale)))

    game_display.blit(surf, dest=(0, 0))
    pygame.display.update()
    clock.tick(fps)

    for i, bot in enumerate(bots):
      timestep_bot = dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=timestep.reward[i],
          discount=timestep.discount,
          observation=timestep.observation[i])

      actions[i], states[i] = bot.step(timestep_bot, states[i])

    timestep = env.step(actions)
    ray.shutdown()


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser(description="Visualization Script for Trained Models")
  
  parser.add_argument(
      "--config_dir",
      type=str,
      help="Directory where your experiment config (params.json) is located",
  )

  parser.add_argument(
      "--policies_dir",
      type=str,
      help="Directory where your trained polcies are located",
  )

  parser.add_argument(
      "--horizon",
      type=int,
      default=200,
      help="No. of environment timesteps to render models",
  )

  args = parser.parse_args()

  render_model(args)
  print("Visualization Complete.")
