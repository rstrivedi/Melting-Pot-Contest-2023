from meltingpot import substrate
from ray.rllib.policy import policy
from baselines.train import make_envs

SUPPORTED_SCENARIOS = [
    'allelopathic_harvest__open_0',
    'allelopathic_harvest__open_1',
    'allelopathic_harvest__open_2',
    'clean_up_2',
    'clean_up_3',
    'clean_up_4',
    'clean_up_5',
    'clean_up_6',
    'clean_up_7',
    'clean_up_8',
    'prisoners_dilemma_in_the_matrix__arena_0',
    'prisoners_dilemma_in_the_matrix__arena_1',
    'prisoners_dilemma_in_the_matrix__arena_2',
    'prisoners_dilemma_in_the_matrix__arena_3',
    'prisoners_dilemma_in_the_matrix__arena_4',
    'prisoners_dilemma_in_the_matrix__arena_5',
    'territory__rooms_0',
    'territory__rooms_1',
    'territory__rooms_2',
    'territory__rooms_3',
]

IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']


def get_experiment_config(args, default_config):
    
    if args.exp == 'pd_arena':
        substrate_name = "prisoners_dilemma_in_the_matrix__arena"
    elif args.exp == 'al_harvest':
        substrate_name = "allelopathic_harvest__open"
    elif args.exp == 'clean_up':
        substrate_name = "clean_up"
    elif args.exp == 'territory_rooms':
        substrate_name = "territory__rooms"
    else:
        raise Exception("Please set --exp to be one of ['pd_arena', 'al_harvest', 'clean_up', \
                        'territory_rooms']. Other substrates are not supported.")

    # Fetch player roles
    player_roles = substrate.get_config(substrate_name).default_player_roles

    if args.downsample:
        scale_factor = 8
    else:
        scale_factor = 1

    params_dict = {

        # resources
        "num_rollout_workers": args.num_workers,
        "num_gpus": args.num_gpus,

        # Env
        "env_name": "meltingpot",
        "env_config": {"substrate": substrate_name, "roles": player_roles, "scaled": scale_factor},


        # training
        "seed": args.seed,
        "rollout_fragment_length": 10,
        "train_batch_size": 400,
        "sgd_minibatch_size": 32,
        "disable_observation_precprocessing": True,
        "use_new_rl_modules": False,
        "use_new_learner_api": False,
        "framework": args.framework,

        # agent model
        "fcnet_hidden": (4, 4),
        "post_fcnet_hidden": (16,),
        "cnn_activation": "relu",
        "fcnet_activation": "relu",
        "post_fcnet_activation": "relu",
        "use_lstm": True,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": False,
        "lstm_cell_size": 2,
        "shared_policy": False,

        # experiment trials
        "exp_name": args.exp,
        "stopping": {
                    #"timesteps_total": 1000000,
                    "training_iteration": 1,
                    #"episode_reward_mean": 100,
        },
        "num_checkpoints": 5,
        "checkpoint_interval": 10,
        "checkpoint_at_end": True,
        "results_dir": args.results_dir,
        "logging": args.logging,

    }

    
    # Preferrable to update the parameters in above dict before changing anything below
    
    run_configs = default_config
    experiment_configs = {}
    tune_configs = None

    # Resources 
    run_configs.num_rollout_workers = params_dict['num_rollout_workers']
    run_configs.num_gpus = params_dict['num_gpus']


    # Training
    run_configs.train_batch_size = params_dict['train_batch_size']
    run_configs.sgd_minibatch_size = params_dict['sgd_minibatch_size']
    run_configs.preprocessor_pref = None
    run_configs._disable_preprocessor_api = params_dict['disable_observation_precprocessing']
    run_configs.rl_module(_enable_rl_module_api=params_dict['use_new_rl_modules'])
    run_configs.training(_enable_learner_api=params_dict['use_new_learner_api'])
    run_configs = run_configs.framework(params_dict['framework'])
    run_configs.log_level = params_dict['logging']
    run_configs.seed = params_dict['seed']

    # Environment
    run_configs.env = params_dict['env_name']
    run_configs.env_config = params_dict['env_config']

    # Setup multi-agent policies. The below code will initialize independent
    # policies for each agent.
    base_env = make_envs.env_creator(run_configs.env_config)
    policies = {}
    player_to_agent = {}
    for i in range(len(player_roles)):
        rgb_shape = base_env.observation_space[f"player_{i}"]["RGB"].shape
        sprite_x = rgb_shape[0]
        sprite_y = rgb_shape[1]

        policies[f"agent_{i}"] = policy.PolicySpec(
            observation_space=base_env.observation_space[f"player_{i}"],
            action_space=base_env.action_space[f"player_{i}"],
            config={
                "model": {
                    "conv_filters": [[16, [8, 8], 1],
                                    [128, [sprite_x, sprite_y], 1]],
                },
            })
        player_to_agent[f"player_{i}"] = f"agent_{i}"

    run_configs.multi_agent(policies=policies, policy_mapping_fn=(lambda agent_id, *args, **kwargs: 
                                                                  player_to_agent[agent_id]))
    
    run_configs.model["fcnet_hiddens"] = params_dict['fcnet_hidden']
    run_configs.model["post_fcnet_hiddens"] = params_dict['post_fcnet_hidden']
    run_configs.model["conv_activation"] = params_dict['cnn_activation'] 
    run_configs.model["fcnet_activation"] = params_dict['fcnet_activation']
    run_configs.model["post_fcnet_activation"] = params_dict['post_fcnet_activation']
    run_configs.model["use_lstm"] = params_dict['use_lstm']
    run_configs.model["lstm_use_prev_action"] = params_dict['lstm_use_prev_action']
    run_configs.model["lstm_use_prev_reward"] = params_dict['lstm_use_prev_reward']
    run_configs.model["lstm_cell_size"] = params_dict['lstm_cell_size']

    # Experiment Trials
    experiment_configs['name'] = params_dict['exp_name']
    experiment_configs['stop'] = params_dict['stopping']
    experiment_configs['keep'] = params_dict['num_checkpoints']
    experiment_configs['freq'] = params_dict['checkpoint_interval']
    experiment_configs['end'] = params_dict['checkpoint_at_end']
    if args.framework == 'tf':
        experiment_configs['dir'] = f"{params_dict['results_dir']}/tf"
    else:
        experiment_configs['dir'] = f"{params_dict['results_dir']}/torch"
 
    return run_configs, experiment_configs, tune_configs