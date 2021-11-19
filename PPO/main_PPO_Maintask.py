
import env_PPO_Maintask as Env
from stable_baselines3 import PPO

env = Env.Ball_env()

LOAD = True    # load the trained model
TRAIN =False   # when training, the UI will be closed
SAVE = False   # save the trained model
DETERM = False # predict is deterministic or not

loadnumber = "./logs/Maintask"
savenumber = "./logs/Maintask"

l_rate = 0.001      # learning rate
n_steps = 9984      # the data buffer stores the state and action
batch_size = 64     # the batch size, train neural network
timesteps = 409600  # training steps

# load weights
if LOAD:
  model=PPO.load(loadnumber,env=env)
  print("load model")
else:
  model = PPO("MlpPolicy", env, verbose=1,device="cpu",
            learning_rate=l_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            
            tensorboard_log="./tensorboard/main/"
            )


if TRAIN:
# train steps, store the weight
  model.learn(total_timesteps=timesteps)
if SAVE:
  model.save(savenumber)
  print("save model")

obs = env.reset()
cnt = 0 # count episode
sucess_cnt = 0 # count success
while(True):
  action, _states = model.predict(obs, deterministic=DETERM)
  obs, reward, done, info = env.step(action)
  env.render()
  if done:
    obs = env.reset()
    cnt+=1
    if reward == 10000: # kick the done
      sucess_cnt+=1
  # print successful rate
    if cnt%10 == 0:
      print("successrate:"+str(sucess_cnt/cnt))
  

