import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from DQN_subtask_env import Ball_env

env = Ball_env()
n_features = 12
n_actions = 9

np.random.seed(1)
tf.compat.v1.set_random_seed(1)

tf.get_logger().setLevel('ERROR')

tf.compat.v1.disable_eager_execution()


class DeepQNetwork(object):

    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')

        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.compat.v1.Session()

        if output_graph:
            tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.cost_his = []

    def store_transition(self, s, a, r, s_):

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1


    def _build_net(self):
        tf.compat.v1.reset_default_graph()

        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions],
                                                 name='Q_target')
        with tf.compat.v1.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.compat.v1.variable_scope('l1'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer,
                                               collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer,
                                               collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.compat.v1.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.compat.v1.squared_difference(self.q_target, self.q_eval))

        with tf.compat.v1.variable_scope('train'):
            optimizer = tf.compat.v1.train.RMSPropOptimizer(self.lr)
            self._train_op = optimizer.minimize(self.loss)

        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')  # 接收下个 observation
        with tf.compat.v1.variable_scope('target_net'):
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer,
                                               collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, self.n_actions], initializer=b_initializer,
                                               collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def choose_action(self, observation):
        observation = np.array(observation)

        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)

        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        self.memory_counter = 10
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.s_: batch_memory[:, -self.n_features:],
                                                  self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        eval_act_index = batch_memory[:, self.n_features].astype(int)

        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1

    def test(self):
        num_test = 100

        # Print route
        f = {}

        # Initialize count, and data store lists
        num_find_goal = 0
        reward_list = []
        steps_list = []

        # run 100 episode to test the correctness of the method
        for i in range(1000):
            # resert the environment
            observation = env.reset()

            for j in range(1000):
                # render the environment
                env.render()

                # Choose the best action based on the optimal_policy
                observation = observation[np.newaxis, :]
                action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action = np.argmax(action_value)

                # perform action and get a tuple
                next_observation, reward, done = env.step(action)

                if done:
                    # Record the number of goal reaching
                    if reward == 1:
                        num_find_goal += 1

                    # While a episode terminates, record the total reward, step
                    # Then add to the list
                    r = reward
                    step = j + 1
                    reward_list += [r]
                    steps_list += [step]

                    break

                observation = next_observation

        print("correctness:{}".format(num_find_goal / num_test))

        # Plot results
        plt.figure()
        plt.plot(np.arange(len(steps_list)), steps_list, 'r')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        plt.figure()
        plt.plot(np.arange(len(reward_list)), reward_list, 'r')
        plt.title('Episode via Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')

        # Showing the plots
        plt.show()
