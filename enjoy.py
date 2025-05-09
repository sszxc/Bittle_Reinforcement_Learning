import time
import torch
import pybullet as p
import numpy as np
np.set_printoptions(precision=2, suppress=False)

from stable_baselines3 import PPO, SAC
from opencat_gym_env import OpenCatGymEnv

env = OpenCatGymEnv(render=True, fixed_target_velocity=[-1.0, 0.0, 0.0], never_end=True)

# ====== load model ======
# directly load the model from .zip file
# model = PPO.load("trained/ckpts/PPO_0506_174547_forward_only/rl_model_64000_steps.zip")
# model = PPO.load("trained/ckpts/PPO_0506_184145/rl_model_40000_steps.zip")
# model = PPO.load("trained/ckpts/PPO_0506_190544/rl_model_880000_steps.zip")
model = PPO.load("trained/ckpts/PPO_0506_211304_diagonally/rl_model_320000_steps.zip")
# model = PPO.load("trained/ckpts/PPO_0506_220135_diagonally_norotate/rl_model_3680000_steps.zip")

# # initialize a PPO model with the same structure, then load the weights
# model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), verbose=0)
# model.policy.load_state_dict(torch.load("trained/ckpts/PPO_0506_232613_diagonally_norotate/policy_weights.pth", map_location='cpu'))
# ====== load model ======

obs, info = env.reset()
# time.sleep(10)
sum_reward = 0
print("\n ===== Start to Run! ===== \n")

for i in range(500):    
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"action: {action}, reward: {reward:.2f}", end="\r")
    sum_reward += reward
    env.render(mode="human")
    time.sleep(0.02)  # add delay to make the rendering more smooth
    if terminated or truncated:
        print("Reward", sum_reward)
        sum_reward = 0
        obs, info = env.reset()
