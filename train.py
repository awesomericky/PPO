# to ignore warning message
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
###########################

import gym
import tensorflow as tf
import numpy as np
import random
import wandb

from nets import PPO_agent

# seed number
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

env_name = 'Hopper-v1'
agent_name = 'PPO'
algo = '{}_{}'.format(agent_name, seed)
save_name = '_'.join(env_name.split('-')[:-1])
save_name = "{}_{}".format(save_name, algo)
agent_args = {'agent_name': agent_name,
            'env_name': save_name,
            'discount_factor': 0.99,
            'loss_type': 20,
            'clipping_limit': 0.2,
            'kl_traget': 0.01,
            'policy_hidden1': 64,
            'policy_hidden2': 64,
            'value_hidden1': 64,
            'value_hidden2': 64,
            'p_lr': 0.0003,
            'v_lr': 0.0003,
            'value_epochs':40,
            'batch_size': 64,
            'kl_coefficient': 3,
            'entropy_coefficient': 1,
            'gae_coefficient': 0.95,
            }

