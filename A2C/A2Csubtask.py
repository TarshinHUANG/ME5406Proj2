# =============================================================================
#   Filename         : A2Csubtask.py
#   Author           : Xue Junyuan
#   Description      : Subtask training.
# =============================================================================

import collections
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from typing import List, Tuple
from A2Csubtaskenv import Ball_env  # Import environment

# Hyper-parameters to be adjusted here ##############################################################
max_episodes = 10000  # End training after this number of episode
max_steps_per_episode = 1000  # End episode after this number of timesteps
num_hl_1 = 40  # Number of first hidden layer
num_hl_2 = 20  # Number of second hidden layer
REACH_COUNT_THRESHOLD = 500  # Accomplish the training when reach count reaches the threshold
gamma = 0.99  # Discount factor for future rewards
#####################################################################################################

# Parameters for saving weights #####################################################################
LOAD_WEIGHT = True  # load trained weight or not
LOAD_WEIGHT_DIR = 'subtask_trained'
SAVE_WEIGHT = True  # save trained weight or not
SAVE_WEIGHT_DIR = 'subtask_trained2'
#####################################################################################################

episode_num = 0
reach_count = 0


class AdvantageActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(self, num_actions: int, num_hidden_1_unit: int, num_hidden_2_unit: int,):
        """Initialize."""
        super().__init__()

        self.hidden1 = layers.Dense(num_hidden_1_unit, activation="relu")
        self.hidden2 = layers.Dense(num_hidden_2_unit, activation="relu")
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h1 = self.hidden1(inputs)
        h2 = self.hidden2(h1)
        return self.actor(h2), self.critic(h2)


# 1. Collecting training data
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""
    # Env is the environment package.
    state, reward, done = env.step(action)
    env.render()  # Enable the UI
    state = np.array(state)
    return state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32)


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    """ make env_step for tf """
    return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32])


def run_episode(initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int)\
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""
    global reach_count

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # action_logits_t = explore_action_probs(action_probs_t, epsilon)
        # action = tf.random.categorical(action_logits_t, 1)[0, 0]
        # action_probs_t = tf.nn.softmax(action_logits_t)

        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        done = tf.cast(done, tf.bool)
        if done:
            print('Reached the ball!!!')
            reach_count += 1
            break
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


# 2. Computing expected returns
def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

    return returns


# 3. The actor-critic loss
def compute_loss(action_probs: tf.Tensor,  values: tf.Tensor,  returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    critic_loss = huber_loss(values, returns)
    print('Actor Loss:', actor_loss.numpy(), 'Critic Loss:', critic_loss.numpy())
    total_loss = actor_loss + critic_loss

    return total_loss


# 4. Defining the training step to update parameters
def episode_train(initial_state: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
               gamma: float, max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""
    with tf.GradientTape() as tape:

        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)
        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # Calculating loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss and applies it to the model's parameters
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


# 5. Run the training loop

env = Ball_env()  # Setup simulation environment

# Create A2C model
A2Cmodel = AdvantageActorCritic(num_actions=9, num_hidden_1_unit=num_hl_1, num_hidden_2_unit=num_hl_2)
if LOAD_WEIGHT:
    A2Cmodel.load_weights(filepath=LOAD_WEIGHT_DIR)
    print('Loaded weight: ', LOAD_WEIGHT_DIR)

eps = np.finfo(np.float32).eps.item()  # Smallest number recognizable by the float.
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)  # Calculate huber loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Setup optimizer
accomplished = False

while not accomplished:
    running_reward = 0  # Real-time averaged reward
    # Deque to keep last 100 episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=100)
    a = []  # Reward List

    # Total episode loop of training
    with tqdm.trange(max_episodes) as t:
        for i in t:
            episode_num = i
            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            episode_reward = episode_train(initial_state, A2Cmodel, optimizer, gamma, max_steps_per_episode)

            episodes_reward.append(episode_reward.numpy())
            running_reward = statistics.mean(episodes_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward.numpy())
            a.append(episode_reward.numpy())

            # Show average episode reward every 50 episodes
            if i % 100 == 0:
                print('\n')
                print(f'Episode {i}: average reward: {running_reward}')
            if reach_count > REACH_COUNT_THRESHOLD:
                accomplished = True
                print('Accomplished!')
                break
            elif reach_count < 10 and i > 500:
                # Reset the NN if not satisfactory in the beginning
                A2Cmodel = AdvantageActorCritic(num_actions=9, num_hidden_1_unit=num_hl_1,
                                                num_hidden_2_unit=num_hl_2)
                print('Bad model, model reset.')
                break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
print('Reached count: ', reach_count)

plt.style.use('seaborn')
TTT = len(a)
plt.plot(np.arange(1, TTT+1, 1), a, '-', color='r')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.show()

if SAVE_WEIGHT:
    A2Cmodel.save_weights(filepath=SAVE_WEIGHT_DIR, overwrite=True, save_format=None, options=None)
    print('Weight saved as: ', SAVE_WEIGHT_DIR)
