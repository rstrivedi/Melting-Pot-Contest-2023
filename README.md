# MeltingPotContest
Official Repository for Melting Pot Contest Experiments

>**NOTE:** This repository provides a basic startup code for training RLLIB agents on Melting Pot.
We plan to add additional support including more examples,
benchmarked configs and GCP compute setup over the next couple of weeks. We will
also make any necessary updates based on feedback from participants. Hence, it is recommended
to sync up your fork every few days for next couple of weeks. We will post in discussion
forums about any updates as they become available.

# Table of Contents
- [Substrates and Scenarios](#substrates-and-scenarios)
- [Installation Guidelines](#installation-guidelines)
- [Run Training](#run-training)
- [Run Evaluation](#run-evaluation)
- [Code Structure](#code-structure)
- [How to Guide](#how-to-guide)
  - [Make submission](#make-submission)
  - [Visualization](#visualization)
  - [Logging](#logging)
- [Identified Issues with Ray 2.6.1](#identified-issues-with-ray-2.6.1)

## Substrates and Scenarios

For this contest, we will focus on following 4 substrates and we list the corresponding
validation scenarios that your submission will be evaluated on during development phase:

| Substrate | Scenarios |
| --------- | --------- |
| allelopathic_harvest__open | allelopathic_harvest__open_0 |
| | allelopathic_harvest__open_1 |
| | allelopathic_harvest__open_2 |
| clean_up | clean_up_2|
| | clean_up_3 |
| | clean_up_4 |
| | clean_up_5 |
| | clean_up_6 |
| | clean_up_7 |
| | clean_up_8 |
| prisoners_dilemma_in_the_matrix__arena | prisoners_dilemma_in_the_matrix__arena_0 |
| | prisoners_dilemma_in_the_matrix__arena_1 |
| | prisoners_dilemma_in_the_matrix__arena_2 |
| | prisoners_dilemma_in_the_matrix__arena_3 |
| | prisoners_dilemma_in_the_matrix__arena_4 |
| | prisoners_dilemma_in_the_matrix__arena_5 |
| territory__rooms | territory__rooms_0 |
| | territory__rooms_1 |
| | territory__rooms_2 |
| | territory__rooms_3 |

## Installation Guidelines

### MacOS Ventura 13.2.1 and Ubuntu 20.04 LTS

The baseline codes and accompanying MeltingPot installation has been tested on MacOS with support for x86_64 platform. If you use newer M1 chips, there may be additional steps required. You are welcome to post in discussion forums if you encounter any issues with installation.

It is recommended to use virtual environments as the setup requires specific versions for some libraries. Below, we provide installation with [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) package manager.

```
git clone <this-repo>
cd <repo-home>
conda create -n mpc_main python=3.10
conda activate mpc_main
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip intall -e .
sh run_patches.sh
```

### Google Cloud Platform Setup

Coming Soon!


## Run Training

```
python baselines/train/run_ray_train.py [OPTIONS]
```
```
OPTIONS:
  -h, --help            show this help message and exit
  --num_workers NUM_WORKERS
                        Number of workers to use for sample collection. Setting it zero will use same worker for collection and model training.
  --num_gpus NUM_GPUS   Number of GPUs to run on (can be a fraction)
  --local               If enabled, init ray in local mode.
  --no-tune             If enabled, no hyper-parameter tuning.
  --algo {ppo}          Algorithm to train agents.
  --framework {tf,torch}
                        The DL framework specifier (tf2 eager is not supported).
  --exp {pd_arena,al_harvest,clean_up,territory_rooms}
                        Name of the substrate to run
  --seed SEED           Seed to run
  --results_dir RESULTS_DIR
                        Name of the wandb group
  --logging {DEBUG,INFO,WARN,ERROR}
                        The level of training and data flow messages to print.
  --wandb WANDB         Whether to use WanDB logging.
  --downsample DOWNSAMPLE
                        Whether to downsample substrates in MeltingPot. Defaults to 8.
  --as-test             Whether this script should be run as a test.

```

## Run Evaluation

```
python baselines/evaluation/evaluate.py [OPTIONS]
```
```
OPTIONS:
  -h, --help            show this help message and exit
  --num_episodes NUM_EPISODES
                        Number of episodes to run evaluation
  --eval_on_scenario EVAL_ON_SCENARIO
                        Whether to evaluate on scenario. If this is False, evaluation is done on substrate
  --scenario SCENARIO   Name of the scenario. This cannot be None when eval_on_scenario is set to True.
  --config_dir CONFIG_DIR
                        Directory where your experiment config (params.json) is located
  --policies_dir POLICIES_DIR
                        Directory where your trained policies are located
  --create_videos CREATE_VIDEOS
                        Whether to create evaluation videos
  --video_dir VIDEO_DIR
                        Directory where you want to store evaluation videos
```
## Code Structure

```
.
├── meltingpot          # A forked version of meltingpot used to train and test the baselines
├── setup.py            # Contains all the information about dependencies required to be installed
└── baselines           # Baseline code to train RLLib agents
    ├── customs         # Add custom policies and metrics here
    |── evaluation      # Evaluate trained models on substrate and scenarios locally
    ├── models          # Add models not registered in Rllib here
    |── tests           # Unit tests to test environment and training
    ├── train           # All codes related to training baselines
      |__configs.py     # Modify model and policy configs in this file
    |── wrappers        # Example code to write wrappers around your environment for added functionality
```

## How to Guide


### Make Submission
---
The trained models will be available in the results folder configured by you.
Please refer to the guidelines on AICrowd [submision-starter-kit](https://gitlab.aicrowd.com/aicrowd/challenges/meltingpot-2023/meltingpot-2023-starter-kit) to make a
submission using the trained checkpoints.

### Visualization
---
#### How to render trained models?

```
python baselines/train/render_models.py [OPTIONS]
```
```
OPTIONS:
  -h, --help            show this help message and exit
  --config_dir CONFIG_DIR
                        Directory where your experiment config (params.json) is located
  --policies_dir POLICIES_DIR
                        Directory where your trained policies are located
  --horizon HORIZON     No. of environment timesteps to render models
```

#### How to visualize scenario plays?

You can also generate videos of agents behavior in various scenarios during local evaluation.
To do this, set `create_videos=True` and `video_dir='<PATH to video directory>'` while running evaluation.
If `eval_on_scenario=False`, this will create video plays of evaluation on substrate.

```
python baselines/evaluation/evaluate.py --create_videos=True --video_dir='' [OPTIONS]
```

**Note:** The script for generating these videos is located in `VideoSubject` class in `meltingpot/utils/evaluation/evaluation.py`. Modify this class to play with video properties such as codec, fps etc. or use different video writer. If you do not use meltingpot code from this repo, we have found that the generated videos are rendered very tiny. To fix that, add `rgb_frame = rgb_frame.repeat(scale, axis=0).repeat(scale, axis=1)` after `line 88` to extrapolate the image, where we used `scale=32`.   

### Logging
---
You can use either Wandb or Tensorboard to log and visualize your training landscape. The install setup provided includes support for both of them.

#### WanDB Logging

To setup Wandb:

1. Create an account on [Wandb](https://wandb.ai) website
2. Get the API key from your account and set corresponding environment variable using `export WANDB_API_KEY=<Your Key>`
3. Enable Wandb logging during training using  `python run_ray_train.py --wandb=True`

#### Tensorboard Logging

To visualize your results with TensorBoard, run: `tensorboard --logdir <results_dir>`


## Identified issues with Ray 2.6.1

During our training, we found issues with both tf and torch backends that leads to errors when using default lstm wrapper provided by rllib. Our installation script above provides fix patches `ray_patch.sh` for the same. But if you use the manual installation approach, the following fixes need to be applied after installation:

*  For tf users:

In your Python library folder, in the file ray/rllib/policy/sample_batch.py, replace line 636 with the following snippet:

```python
time_lengths = tree.map_structure(lambda x: len(x), data[next(iter(data))])
flattened_lengths = tree.flatten(time_lengths)
assert all(t == flattened_lengths[0] for t in flattened_lengths)
data_len = flattened_lengths[0]
```

* For torch users:

In your Python library folder, in the file ray/rllib/models/torch/complex_input_net.py replace line 181 with:

```python
self.num_outputs = concat_size if not self.post_fc_stack else self.post_fc_stack.num_outputs
```
