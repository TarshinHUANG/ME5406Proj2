Readme
This project is built for #ME5406 Deep Learning for Robotics @NUS

This project utilized Pytorch and Tensorflow so that the learning agent learns firstly approaching to the football and then kicking the football to the target position. Deep RL methods including DQN, DDPG, A2C, and PPO are attempted in this project, and different methods has different environment requirements. The introduction of included files and methods to setup the environments with respect to different learning method are introduced as below:

DQN

Files included:

createui.py: the UI controlling by keybroads
DQN_maintask_training_test.py: run to train and test the main task.
DQN_maintask_RL_main.py: rnetwork the main task.
DQN_maintask_env.py: environment for the main task.
DQN_subtask_training_test.py: run to train and test the subtask.
DQN_subtask_RL_main.py: The network the subtask.
DQN_subtask_env.py: environment for the subtask.

'gate.png', 'robot.png', 'soccer.png': Three figures for the UI.


Setup procedure

1. Put all the source codes, model files, and picture files in a project folder with Python3 interpreter.
2. Make up the environment based on 'requirement.txt'.
3. Directly run any python file for training or testing. The purposes of the files are already indicated in their names. The adjustable parameters and booleans are listed in the beginning of the code.