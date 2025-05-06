import time
import pybullet as p
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv

env = OpenCatGymEnv()
model = PPO.load("trained/PPO_0506_145436_backward")

env.set_target_velocity(forward_velocity=1.0, lateral_velocity=0.0, angular_velocity=0.0)
obs, info = env.reset()
sum_reward = 0

for i in range(500):    
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    sum_reward += reward
    env.render(mode="human")
    # time.sleep(0.01)  # 添加10ms延时使渲染更流畅
    if terminated or truncated:
        print("Reward", sum_reward)
        sum_reward = 0
        obs, info = env.reset()
