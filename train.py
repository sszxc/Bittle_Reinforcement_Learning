import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from opencat_gym_env import OpenCatGymEnv
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__ == "__main__":
    exp_name = f"PPO_{datetime.now().strftime('%m%d_%H%M%S')}_diagonally_norotate"

    # Set up number of parallel environments
    parallel_env = 4
    env = make_vec_env(OpenCatGymEnv,
                       n_envs=parallel_env, 
                       vec_env_cls=SubprocVecEnv)
    # Single environment for testing
    # env = OpenCatGymEnv()
    # check_env(env)

    # Change architecture of neural network to two hidden layers of size 256
    custom_arch = dict(net_arch=[256, 256])

    # define the checkpoint callback
    os.makedirs(f'trained/ckpts/{exp_name}/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=f'trained/ckpts/{exp_name}/',  # save path
        # name_prefix='ppo_model',  # file name prefix
        save_replay_buffer=True,
        save_vecnormalize=True,  # for vectorized environments
    )

    # ====== define PPO agent ======
    # from scratch
    model = PPO('MlpPolicy', env, seed=42, 
                policy_kwargs=custom_arch, 
                n_steps=int(2048*8/parallel_env), verbose=1,
                tensorboard_log="trained/tensorboard_logs/")
    # load ckpt
    # model = PPO.load("trained/ckpts/PPO_0506_224211_diagonally_norotate/rl_model_4960000_steps.zip", 
    #                   env, policy_kwargs=custom_arch, 
    #                   n_steps=int(2048*8/parallel_env), verbose=1, 
    #                   tensorboard_log="trained/tensorboard_logs/")
    # ====== define PPO agent ======

    model.learn(5e6, tb_log_name=exp_name, callback=checkpoint_callback)

    # save the policy weights in PyTorch format (more general)
    torch.save(model.policy.state_dict(), f"trained/ckpts/{exp_name}/policy_weights.pth")
