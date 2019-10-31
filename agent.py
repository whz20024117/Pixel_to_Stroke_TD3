from stable_baselines.td3 import TD3, LnCnnPolicy, LnMlpPolicy
from env import *
import tensorflow as tf
from config import config


policy = LnMlpPolicy
#env = SketchDesigner(SketchDiscriminator(config['SAVED_GAN']))
env = SketchDesigner(SketchClassifier(config['SAVED_CNN']))
agent = TD3(policy, env, random_exploration=0.1, tensorboard_log='./log/',verbose=1)
#agent.get_env().env_method('get_policy', agent.policy_tf)
agent.get_env().get_policy(agent.policy_tf)


for _ in range(200):
    agent.learn(1000, reset_num_timesteps=False)
    agent.save('./save/model/1')

