# Football Court

This project utilized Pytorch and Tensorflow so that the learning agent learns firstly approaching to the football and then kicking the football to the target position. Deep RL methods including DQN, DDPG, A2C, and PPO are attempted in this project, and different methods has different environment requirements. The introduction of included files and methods to setup the environments with respect to different learning method are introduced as below:

## The A2C Algorithm

### Setup procedure

1. Put all the source codes, model files, and picture files in a project folder with Python3 interpreter.
2. Make up the environment based on 'requirement.txt'.
3. Directly run any python file for training or testing. The purposes of the files are already indicated in their names. The adjustable parameters and booleans are listed in the beginning of the code.

### The file structure

> ./A2C/
> 	A2Cmaintask_test.py							: run to test the main task.
> 	A2Cmaintask_train.py						   : run to train the main task.
> 	A2Cmaintaskenv.py							  : environment for the main task.
> 	A2Csubtask_test.py							   : run to test the subtask.
> 	A2Csubtask_train.py							 : run to train the subtask.
> 	A2Csubtaskenv.py								  : environment for the subtask.
>
> ​	'gate.png', 'robot.png', 'soccer.png'	: Three figures for the UI.
>
> ​	'checkpoint'											: format file for Tensorflow model.
> ​	'maintask_trained.data-00000-of-00001', 'maintask_trained.index': Saved model for 	the main task.
> ​	'subtask_trained.data-00000-of-00001', 'subtask_trained.index': Saved model for the subtask.

## The PPO Algorithm

### Environment build

​	For the PPO algorithm, the Deep learning framework is Pytorch. To run the program, you need to run the code below to install necessary library.

```bash
pip install stable-baselines3[extra]
pip install gym==0.19.0
pip install pyglet==1.5.21
pip install numpy==1.19.5
pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -U torch-tb-profiler
```

### Build and run file

After installing the library, run the code below to see the results of trained model.

```bash
python main_PPO_Maintask
python main_PPO_Subtask
```

If want to retrain the model, then open these two main.py files, change the boolean value TRAIN into True.

### The file structure

> ./PPO/                                                   
> 	env_PPO_Maintask.py                    	   : The env for the Maintask
> 	env_PPO_Subtask.py              			    : The env for the Subtask
> 	main_PPO_Maintask.py					     : The main for the Maintask
> 	main_PPO_Subtask.py						   : The main for the Subtask
> 	requirements.txt								    : The requirements for the python enviroments
> 	gate.png												   : The picture for the env render
> 	robot.png												 : The picture for the env render
> 	soccer.png												: The picture for the env render
> 	logs/
> 		Maintask.zip										 : The trained model for the Maintask
> 		Subtask.zip										   : The trained model for the Subtask

