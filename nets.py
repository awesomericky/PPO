import numpy as np
import tensorflow as tf
import pickle
from copy import deepcopy
import os
import sys
import pdb

class PPO_agent:
    def __init__(self, env, args):
        """
        [Available Loss Type]
        10 : 'naive loss' w/o 'entropy loss'
        20 : 'naive loss' w/ 'clipping' w/o 'entropy loss'
        30 : 'naive loss' w/ 'fixed KL penalty' w/o 'entropy loss'
        40 : 'naive loss' w/ 'adaptive KL penalty' w/o 'entropy loss'
        11 : 'naive loss' w/ 'entropy loss'
        21 : 'naive loss' w/ 'clipping' w/ 'entropy loss'
        31 : 'naive loss' w/ 'fixed KL penalty' w/ 'entropy loss'
        41 : 'naive loss' w/ 'adaptive KL penalty' w/ 'entropy loss'
        """
        self.env = env
        self.name = args['agent_name']
        self.checkpoint_dir = '{}/checkpoint'.format(args['env_name'])
        self.discount_factor = args['discount_factor']
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.shape[0]
            self.action_bound_min = env.action_space.low
            self.action_bound_max = env.action_space.high
        except:
            self.action_dim = 1
            self.action_bound_min = -1.0
            self.action_bound_max = 1.0
        self.loss_type = args['loss_type']
        self.clipping_limit = args['clipping_limit']
        self.kl_target = args['kl_target']
        self.policy_hidden1_units = args['policy_hidden1']
        self.policy_hidden2_units = args['policy_hidden2']
        self.value_hidden1_units = args['value_hidden1']
        self.value_hidden2_units = args['value_hidden2']
        self.p_lr = args['p_lr']
        self.v_lr = args['v_lr']
        self.value_epochs = args['value_epochs']
        self.batch_size = args['batch_size']
        self.kl_coeff = args['kl_coefficient']
        self.entropy_coeff = args['entropy_coefficient']
        self.gae_coeff = args['gae_coefficient']
        self.EPS = np.finfo('float').eps

        with tf.variable_scope(self.name):
            # placeholder
            self.states = tf.placeholder(tf.float32, [None, self.state_dim], name='states')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='actions')
            self.v_targets = tf.placeholder(tf.float32, [None, ], name='value_targets')
            self.gaes = tf.placeholder(tf.float32, [None, ], name='gaes')
            self.old_mean = tf.placeholder(tf.float32, [None, self.action_dim], name='old_mean')
            self.old_std = tf.placeholder(tf.float32, [None, self.action_dim], name='old_std')
            self.log_prob_old = tf.placeholder(tf.float32, [None, ], name='log_prob_old')

            # policy & value & KL
            self.mean, self.std = self.build_policy_model('policy')
            self.value = self.build_value_model('value')
            self.kl, self.entropy = self.get_kl_divergence_and_entropy()

            # action
            self.feature_action = self.mean + tf.multiply(tf.random_normal(tf.shape(self.mean)), self.std)
            self.sample_noise_actions = self.unnormalize_action(self.feature_action)
            self.sample_actions = self.unnormalize_action(self.mean)
            norm_actions = self.normalize_action(self.actions)

            # policy loss & optimizer
            p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
            self.log_prob = -tf.reduce_sum(tf.log(self.std + self.EPS) + 0.5*np.log(2*np.pi) + tf.squared_difference(norm_actions, self.mean)/(2*tf.square(self.std) + self.EPS), axis=1)  ##(num_data, )
            prob_ratio = tf.exp(self.log_prob - self.log_prob_old)
            if self.loss_type == 10:
                self.objective = -tf.reduce_mean(prob_ratio*self.gaes)  ## (data, ) == step??? gaes dimension???
            elif self.loss_type == 20:
                term_1 = prob_ratio*self.gaes
                term_2 = tf.clip_by_value(prob_ratio, clip_value_min=1-self.clipping_limit, clip_value_max=1+self.clipping_limit)*self.gaes
                self.objective = -(tf.reduce_mean(tf.reduce_min(tf.concat([term_1, term_2], axis=0), axis=0)))
            elif (self.loss_type == 30) or (self.loss_type == 40):
                self.objective = -(tf.reduce_mean(prob_ratio*self.gaes) - self.kl_coeff*self.kl)
            elif self.loss_type == 11:
                self.objective = -(tf.reduce_mean(prob_ratio*self.gaes) + self.entropy_coeff*self.entropy)
            elif self.loss_type == 21:
                term_1 = prob_ratio*self.gaes
                term_2 = tf.clip_by_value(prob_ratio, clip_value_min=1-self.clipping_limit, clip_value_max=1+self.clipping_limit)*self.gaes
                self.objective = -(tf.reduce_mean(tf.reduce_min(tf.concat([term_1, term_2], axis=0), axis=0)) + self.entropy_coeff*self.entropy)
            elif (self.loss_type == 31) or (self.loss_type == 41):
                self.objective = -(tf.reduce_mean(prob_ratio*self.gaes) - self.kl_coeff*self.kl + self.entropy_coeff*self.entropy)
            else:
                raise ValueError('Not available loss type input')
            p_optimizer = tf.train.AdamOptimizer(learning_rate=self.p_lr)
            self.p_grads = tf.gradients(self.objective, p_vars)
            self.p_train_op = p_optimizer.apply_gradients(zip(self.p_grads, p_vars))

            # value_loss & optimizer
            v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
            self.v_loss = 0.5*tf.square(self.v_targets - self.value)
            self.v_loss = tf.reduce_mean(self.v_loss)
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.v_lr)
            self.v_grads = tf.gradients(self.v_loss, v_vars)
            self.v_train_op = v_optimizer.apply_gradients(zip(self.v_grads, v_vars))
        
        # make session and load model
        config = tf.ConfigProto()
        # config.gpu.options.allow_growth = True  ## uncomment it if using GPU is available
        self.sess = tf.Session(config=config)
        self.load()
    
    def build_policy_model(self, name='policy', reuse=False):
        param_initializer = lambda : tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope(name, reuse=reuse):
            inputs = self.states
            model = tf.layers.dense(inputs, self.policy_hidden1_units, activation=None, kernel_initializer=param_initializer(), bias_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.policy_hidden2_units, activation=None, kernel_initializer=param_initializer(), bias_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            mean = tf.layers.dense(model, self.action_dim, activation=tf.tanh, kernel_initializer=param_initializer(), bias_initializer=param_initializer())
            logits_std = tf.get_variable('logits_std', shape=(self.action_dim), initializer=tf.random_normal_initializer(mean=-1.0, stddev=0.01), trainable=True)
            std = tf.ones_like(mean)*tf.nn.softplus(logits_std)
        return mean, std  # (num_steps, action_dim), (num_steps, action_dim)
    
    def build_value_model(self, name='value', reuse=False):
        param_initializer = lambda : tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope(name, reuse=reuse):
            inputs = self.states
            model = tf.layers.dense(inputs, self.value_hidden1_units, activation=None, kernel_initializer=param_initializer(), bias_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.value_hidden2_units, activation=None, kernel_initializer=param_initializer(), bias_initializer=param_initializer())
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, 1, activation=None, kernel_initializer=param_initializer(), bias_initializer=param_initializer())
            outputs = tf.reshape(model, [-1])  ## WHY????
        return outputs  # (num_steps,)
    
    def get_kl_divergence_and_entropy(self):
        mean, std = self.mean, self.std
        old_mean, old_std = self.old_mean, self.old_std
        old_log_std = tf.log(old_std + self.EPS)
        new_log_std = tf.log(std + self.EPS)
        frac_std_old_new = old_std/(std + self.EPS)
        kl = tf.reduce_mean(tf.reduce_sum(new_log_std - old_log_std + 0.5*tf.square(frac_std_old_new) + 0.5*tf.square((old_mean - mean)/(std + self.EPS))- 0.5, axis=-1))
        entropy = tf.reduce_mean(tf.reduce_sum(new_log_std + 0.5 + 0.5*np.log(2*np.pi), axis=-1))
        return kl, entropy

    def normalize_action(self, actions):
        # normalizing actions in range between -1 and 1
        temp_a = 2.0/(self.action_bound_max - self.action_bound_min)
        temp_b = (self.action_bound_max + self.action_bound_min)/(self.action_bound_max - self.action_bound_min)
        temp_a = tf.ones_like(actions)*temp_a
        temp_b = tf.ones_like(actions)*temp_b
        return temp_a*actions - temp_b
    
    def unnormalize_action(self, feature_actions):
        # unnormalizing feature actions(outputs from policy network which are in range between -1 and 1)
        # in range between action_bound_min and action_bound_max
        temp_a = (self.action_bound_max - self.action_bound_min)/2.0
        temp_b = (self.action_bound_max + self.action_bound_min)/2.0
        temp_a = tf.ones_like(feature_actions)*temp_a
        temp_b = tf.ones_like(feature_actions)*temp_b
        return temp_a*feature_actions + temp_b

    def get_action(self, state, is_train):
        state = state.reshape((1, len(state)))
        if is_train:
            [[action], [value]] = self.sess.run([self.sample_noise_actions, self.value], feed_dict={self.states:state})
        else:
            [[action], [value]] = self.sess.run([self.sample_actions, self.value], feed_dict={self.states:state})
        clipped_action = np.clip(action, self.action_bound_min, self.action_bound_max)
        return clipped_action, value
    
    def get_gaes_targets(self, rewards, values, next_values):
        # computing truncated version of GAE (considering only 'gae_truncated_steps' number of steps)
        TD_errors = np.array(rewards) + self.discount_factor*np.array(next_values) - np.array(values)
        gaes = deepcopy(TD_errors)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] += self.discount_factor*self.gae_coeff*gaes[t+1]
        targets = np.array(values) + np.array(gaes)
        return gaes, targets
    
    def train(self, trajs):
        states = np.array(trajs[0], dtype=np.float32)  # (num_steps, state_dim)
        actions = np.array(trajs[1], dtype=np.float32)  # (num_steps, action_dim)
        v_targets = np.array(trajs[2], dtype=np.float32)[:, np.newaxis]  # (num_steps,)
        gaes = np.array(trajs[3], dtype=np.float32)[:, np.newaxis]  # (num_steps,)
        total_trajs = np.concatenate((states, actions, v_targets, gaes), axis=1)  # (num_steps, state_dim + action_dim + 2)

        # update value network
        for i in range(self.value_epochs):
            np.random.shuffle(total_trajs)
            states = total_trajs[:, :self.state_dim]
            v_targets = total_trajs[:, self.state_dim+self.action_dim]
            start = 0
            while start < total_trajs.shape[0] - self.batch_size:
                end = start + self.batch_size
                batch_states = states[start:end, :]
                batch_v_targets = v_targets[start:end]

                self.sess.run([self.v_train_op], feed_dict={
                    self.states: batch_states,
                    self.v_targets: batch_v_targets
                })
                start += self.batch_size
        
        # preprocessing data
        np.random.shuffle(total_trajs)
        states = total_trajs[:, :self.state_dim]
        actions = total_trajs[:, self.state_dim:self.state_dim+self.action_dim]
        v_targets = total_trajs[:, self.state_dim+self.action_dim]
        gaes = total_trajs[:, self.state_dim+self.action_dim+1]
        [old_means, old_stds, old_log_probs] = self.sess.run([self.mean, self.std, self.log_prob], feed_dict={self.states: states, self.actions: actions})

        # update policy network
        start = 0
        while start < total_trajs.shape[0] - self.batch_size:
            end = start + self.batch_size
            batch_states = states[start:end, :]
            batch_actions = actions[start:end, :]
            batch_gaes = gaes[start:end]
            batch_old_means = old_means[start:end, :]
            batch_old_stds = old_stds[start:end, :]
            batch_old_log_probs = old_log_probs[start:end]
            
            self.sess.run([self.p_train_op], feed_dict={
                self.states: batch_states,
                self.actions: batch_actions,
                self.old_mean: batch_old_means,
                self.old_std: batch_old_stds,
                self.log_prob_old: batch_old_log_probs,
                self.gaes: batch_gaes
            })
            start += self.batch_size

        # ubdate KL penalty coefficient
        if (self.loss_type == 40) or (self.loss_type == 41):
            kl = self.sess.run(self.kl, feed_dict={
                self.states: states,
                self.actions: actions,
                self.old_mean: old_means,
                self.old_std: old_stds
            })
            if kl < self.kl_target/1.5:
                self.kl_coeff = self.kl_coeff/2.0
            elif kl > self.kl_target*1.5:
                self.kl_coeff = self.kl_coeff*2.0
            else:
                pass
        
        # return result of training
        objective = self.sess.run(self.objective, feed_dict={
            self.states: states,
            self.actions: actions,
            self.gaes: gaes,
            self.old_mean: old_means,
            self.old_std: old_stds,
            self.log_prob_old: old_log_probs
        })
        v_loss = self.sess.run(self.v_loss, feed_dict={
            self.states: states,
            self.v_targets: v_targets
        })
        kl = self.sess.run(self.kl, feed_dict={
            self.states: states,
            self.actions: actions,
            self.old_mean: old_means,
            self.old_std: old_stds
        })
        return objective, v_loss, kl

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        print('Model saved')
    
    def load(self):
        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))
        
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Sucess to load model')
        else:
            self.sess.run(tf.global_variables_initializer())
            print('Fail to load model')