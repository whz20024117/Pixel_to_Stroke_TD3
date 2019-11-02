from stable_baselines.td3 import TD3, LnCnnPolicy, LnMlpPolicy
#from stable_baselines.ddpg import DDPG, LnMlpPolicy
from env import *
import tensorflow as tf
from config import config
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.vec_env import DummyVecEnv


policy = LnMlpPolicy
action_noise = NormalActionNoise(mean=np.zeros(config['ACTION_DIM']), sigma=0.1 * np.ones(config['ACTION_DIM']))
#env = SketchDesigner(SketchDiscriminator(config['SAVED_GAN']))
env = SketchDesigner(SketchClassifier(config['SAVED_CNN']))

#env = DummyVecEnv([lambda: env])

agent = TD3(policy, env,
            random_exploration=0.2,
            #action_noise=action_noise,
            #tensorboard_log='./log/',
            verbose=1)
#agent.get_env().env_method('get_policy', agent.policy_tf)
agent.get_env().get_policy(agent.policy_tf)


for _ in range(400):
    agent.learn(1000, reset_num_timesteps=False)
    agent.save('./save/4/model')

