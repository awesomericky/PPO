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
import sys
import os
import wandb

from nets import PPO_agent

# seed number
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

env_name = 'Hopper-v2'
agent_name = 'PPO'
algo = '{}_{}'.format(agent_name, seed)
save_name = '_'.join(env_name.split('-')[:-1])
save_name = "{}_{}".format(save_name, algo)
agent_args = {'agent_name': agent_name,
            'env_name': save_name,
            'discount_factor': 0.99,
            'loss_type': 20,
            'clipping_limit': 0.2,
            'kl_target': 0.01,
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

def train():
    # set environment and agent
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = PPO_agent(env, agent_args)

    # initialize wandb logger
    wandb_name = 'loss' + '_' + str(agent_args['loss_type']) +'_' + 'seed' + '_' + str(seed)
    wandb_project = env_name+'_'+agent_name
    # wandb.init(name=wandb_name, project=wandb_project)

    # training constants
    max_steps = 2048*2
    max_ep_len = 2048
    episodes = int(max_steps/max_ep_len)
    epochs = 50
    save_freq = 10
    accum_total_steps = 0

    for epoch in range(epochs):
        print("="*20)
        print("Epoch : {}".format(epoch+1))
        states = []
        actions = []
        v_targets = []
        gaes = []
        total_steps = 0
        while total_steps < max_steps:
            state = env.reset()
            done = False
            score = 0
            step = 0
            temp_rewards = []
            values = []
            while True:
                step += 1
                assert env.observation_space.contains(state)
                action, value = agent.get_action(state, True)
                assert env.action_space.contains(action)
                next_state, reward, done, info = env.step(action)

                states.append(state)
                actions.append(action)
                temp_rewards.append(reward)
                values.append(value)

                state = next_state
                score += reward

                if done or step >= max_ep_len:
                    break
        
            total_steps += step
            accum_total_steps += step
            if step >= max_ep_len:
                _, value = agent.get_action(state, True)
            else:
                value = 0
            next_values = values[1:] + [value]
            temp_gaes, temp_v_targets = agent.get_gaes_targets(temp_rewards, values, next_values)
            v_targets += list(temp_v_targets)
            gaes += list(temp_gaes)
            # wandb.log({'step': accum_total_steps, 'score': score})
        
        trajs = [states, actions, v_targets, gaes]
        objective, v_loss, kl = agent.train(trajs)
        print('Objective: {} | V_loss: {} | KL: {}'.format(objective, v_loss, kl))

def test():
    # set environment and agent
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = PPO_agent(env, agent_args)
    episodes = int(20)
    max_ep_len = 5000
    for episode in range(episodes):
        state = env.reset()
        done = False
        step = 0
        score = 0
        while not done and step <= max_ep_len:
            step += 1
            assert env.observation_space.contains(state)
            action, value = agent.get_action(state, False)
            assert env.action_space.contains(action)
            next_state, reward, done, info = env.step(action)

            score += reward
            state = next_state
            env.render()
        print("Episode: {} | Score: {}".format(episode, score))
    print('='*10 + ' Test finished ' + '='*10)

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        test()
    else:
        raise ValueError('Not available input for executing train.py')
