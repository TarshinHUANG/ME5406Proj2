import math

import numpy as np
from matplotlib import pyplot as plt

from DQN_subtask_env import Ball_env

ON_TRAIN = True
env = Ball_env
n_features = 12
n_actions = 9
from DQN_subtask_RL_main import DeepQNetwork

ON_TRAIN = True
env = Ball_env
n_features = 12
n_actions = 9

if __name__ == "__main__":
    env = Ball_env()
    RL = DeepQNetwork(n_features=12,
                      n_actions=9,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    steps = []
    # Resulted list for the plotting Episodes via cost
    all_costs = []
    # Resulted list for the plotting Episodes via average accuracy
    accuracy = []
    # List for average rewards
    Reward_list = []

    # Initialize variable
    goal_count = 0
    rewards = 0
    positive_count = 0
    negative_count = 0
    # i is the episode
    for i in range(300):
        # Initial Observation
        observation = env.reset()

        # Initialize step count
        step = 0

        # Initialize cost count
        cost = 0

        # Calculate the accuracy for every 50 steps
        if i != 0 and i % 50 == 0:
            goal_count = goal_count / 50
            accuracy += [goal_count]
            goal_count = 0

        # Record Q value for specific grid for checking converging

        while True:
            # Render environment
            # env.render()

            # RL chooses action based on epsilon greedy policy
            action = RL.choose_action(observation)

            # Takes an action and get the next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # Count the number of Steps in the current Episode
            step += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                # Record the positive cost and negative cost
                if reward > 0:
                    positive_count += 1
                else:
                    negative_count += 1

                # Record the step
                steps += [step]

                # Record the cost
                all_costs += [RL.cost_his]

                # goal count +1, if reaching the goal
                if reward == 1:
                    goal_count += 1

                # Record total rewards to calculate average rewards
                rewards += reward
                Reward_list += [rewards / (i + 1)]

                break

        print('episode:{}'.format(i))

        all_cost_bar = [positive_count, negative_count]
        print(all_cost_bar)
        # Record the data to the list
        all_cost_bar = [positive_count, negative_count]
        print(all_cost_bar)
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.show()

        plt.figure()
        plt.plot(np.arange(len(all_costs)), all_costs, 'b')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.show()

        plt.figure()
        plt.plot(np.arange(len(accuracy)), accuracy, 'b')
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.show()

        plt.figure()
        list = ['Success', 'Fail']
        color_list = ['blue', 'red']
        plt.bar(np.arange(len(all_cost_bar)), all_cost_bar, tick_label=list, color=color_list)
        plt.title('Bar/Success and Fail')
        plt.ylabel('Number')

        plt.figure()
        plt.plot(np.arange(len(Reward_list)), Reward_list, 'b')
        plt.title('Episode via Average rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average rewards')
        plt.show()

RL.test()
