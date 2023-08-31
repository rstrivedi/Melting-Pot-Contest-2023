import argparse
import ray

from absl.testing import absltest
from baselines.train import make_envs
from baselines.train.configs import get_experiment_config
from ray import air
from ray import tune
from ray.tune import registry
from ray.rllib.algorithms import ppo

def get_cli_args():
  
  parser = argparse.ArgumentParser(description="Training Script for Multi-Agent RL in Meltingpot")
  
  parser.add_argument(
      "--num_workers",
      type=int,
      default=0,
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
      "--wandb_project",
      type=str,
      default="wdb_pr",
      help="Name of the wandb project",
  )
  parser.add_argument(
      "--wandb_group",
      type=str,
      default="wdb_gr",
      help="Name of the wandb group",
  )
  parser.add_argument(
      "--results_dir",
      type=str,
      default="./test_train_results",
      help="Name of the wandb group",
  )
  parser.add_argument(
        "--logging",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="DEBUG",
        help="The level of training and data flow messages to print.",
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

class TrainingTests(absltest.TestCase):
  """Tests for run_ray_train with small configuration."""

  def test_training(self):
    args = get_cli_args()

    # Set up Ray. Use local mode for debugging. Ignore reinit error.
    ray.init(local_mode=args.local, ignore_reinit_error=True)

    # Register meltingpot environment
    registry.register_env("meltingpot", make_envs.env_creator)

    # Initialize default configurations for native RLlib algorithms
    trainer = "PPO"
    default_config = ppo.PPOConfig()

    # Fetch experiment configurations
    configs, exp_config, _ = get_experiment_config(args, default_config)
  
    configs.num_gpus=0
    configs.num_rollout_workers=1
    configs.rollout_fragment_length=10
    configs.train_batch_size=400
    configs.sgd_minibatch_size=32
    configs.fcnet_hiddens=(4,4)
    configs.post_fcnet_hiddens=(4,)
    configs.lstm_cell_size=2

    ckpt_config = None

    # Run Trials
    results = tune.Tuner(
        trainer,
        param_space=configs.to_dict(),
        run_config=air.RunConfig(name = exp_config['name'], local_dir=exp_config['dir'], 
                                stop=exp_config['stop'], checkpoint_config=ckpt_config, verbose=0),
    ).fit()

    ray.shutdown()
    self.assertEqual(results.num_errors, 0)
    print(results)

if __name__ == "__main__":
  absltest.main()