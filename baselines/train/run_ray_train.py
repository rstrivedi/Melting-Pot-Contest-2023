import argparse
import os
import ray


from typing import *
from ray import air
from ray import tune
from configs import get_experiment_config
from ray.rllib.algorithms import ppo
from ray.tune import registry
from ray.air.integrations.wandb import WandbLoggerCallback
from baselines.train import make_envs

def get_cli_args():
  
  parser = argparse.ArgumentParser(description="Training Script for Multi-Agent RL in Meltingpot")
  
  parser.add_argument(
      "--num_workers",
      type=int,
      default=2,
      help="Number of workers to use for sample collection. Setting it zero will use same worker for collection and model training.",
  )
  parser.add_argument(
      "--num_gpus",
      type=int,
      default=0,
      help="Number of GPUs to run on (can be a fraction)",
  )
  parser.add_argument(
      "--local",
      action="store_true",
      help="If enabled, init ray in local mode.",
  )
  parser.add_argument(
      "--no-tune",
      action="store_true",
      help="If enabled, no hyper-parameter tuning.",
  )
  parser.add_argument(
        "--algo",
        choices=["ppo"],
        default="ppo",
        help="Algorithm to train agents.",
  )
  parser.add_argument(
        "--framework",
        choices=["tf", "torch"],
        default="torch",
        help="The DL framework specifier (tf2 eager is not supported).",
  )
  parser.add_argument(
      "--exp",
      type=str,
      choices = ['pd_arena','al_harvest','clean_up','territory_rooms'],
      default="pd_arena",
      help="Name of the substrate to run",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=123,
      help="Seed to run",
  )
  parser.add_argument(
      "--results_dir",
      type=str,
      default="./results",
      help="Name of the wandb group",
  )
  parser.add_argument(
        "--logging",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help="The level of training and data flow messages to print.",
  )
  
  parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Whether to use WanDB logging.",
  )

  parser.add_argument(
        "--downsample",
        type=bool,
        default=True,
        help="Whether to downsample substrates in MeltingPot. Defaults to 8.",
  )

  parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test.",
  )

  args = parser.parse_args()
  print("Running trails with the following arguments: ", args)
  return args



if __name__ == "__main__":

  args = get_cli_args()

  # Set up Ray. Use local mode for debugging. Ignore reinit error.
  ray.init(local_mode=args.local, ignore_reinit_error=True)

  # Register meltingpot environment
  registry.register_env("meltingpot", make_envs.env_creator)

  # Initialize default configurations for native RLlib algorithms
  if args.algo == "ppo":
     trainer = "PPO"
     default_config = ppo.PPOConfig()
  else:
     print('The selected option is not tested. You may encounter issues if you use the baseline \
           policy configurations with non-tested algorithms')

  # Fetch experiment configurations
  configs, exp_config, tune_config = get_experiment_config(args, default_config)
  
  # Ensure GPU is available if set to True
  if configs.num_gpus > 0:
     import torch
     if torch.cuda.is_available():
        print("Using GPU device.")
     else:
        print("Either GPU is not available on this machine or not visible to this run. Training using CPU only.")
        configs.num_gpus = 0


  # Setup WanDB 
  if "WANDB_API_KEY" in os.environ and args.wandb:
    wandb_project = f'{args.exp}_{args.framework}'
    wandb_group = "meltingpot"

    # Set up Weights And Biases logging if API key is set in environment variable.
    wdb_callbacks = [
        WandbLoggerCallback(
            project=wandb_project,
            group=wandb_group,
            api_key=os.environ["WANDB_API_KEY"],
            log_config=True,
        )
    ]
  else:
    wdb_callbacks = []
    print("WARNING! No wandb API key found, running without wandb!")


  # Setup hyper-parameter optimization configs here
  if not args.no_tune:
    # NotImplementedError
    tune_config = None
  else:
    tune_config = tune.TuneConfig(reuse_actors=False)


  # Setup checkpointing configurations
  ckpt_config = air.CheckpointConfig(num_to_keep=exp_config['keep'], checkpoint_frequency=exp_config['freq'], 
                                     checkpoint_at_end=exp_config['end'])

  # Run Trials
  results = tune.Tuner(
      trainer,
      param_space=configs.to_dict(),
      run_config=air.RunConfig(name = exp_config['name'], callbacks=wdb_callbacks, local_dir=exp_config['dir'], 
                               stop=exp_config['stop'], checkpoint_config=ckpt_config, verbose=0),
  ).fit()

  best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
  print(best_result)
  
  ray.shutdown()

