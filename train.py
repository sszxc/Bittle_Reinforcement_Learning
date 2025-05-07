import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from opencat_gym_env import OpenCatGymEnv
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback


# Create OpenCatGym environment from class and check if structure is correct
#env = OpenCatGymEnv()
#check_env(env)


if __name__ == "__main__":
    exp_name = f"PPO_{datetime.now().strftime('%m%d_%H%M%S')}"

    # Set up number of parallel environments
    parallel_env = 32
    # env = make_vec_env(OpenCatGymEnv,
    #                    n_envs=parallel_env, 
    #                    vec_env_cls=SubprocVecEnv)
    # Single environment for testing
    env = OpenCatGymEnv()

    # Change architecture of neural network to two hidden layers of size 256
    custom_arch = dict(net_arch=[256, 256])

    # 定义 checkpoint 回调
    os.makedirs(f'trained/ckpts/{exp_name}/', exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f'trained/ckpts/{exp_name}/',  # 保存路径
        # name_prefix='ppo_model',  # 文件名前缀
        save_replay_buffer=True,
        save_vecnormalize=True,  # 向量化的环境下需要用这个
    )

    # Define PPO agent from scratch
    model = PPO('MlpPolicy', env, seed=42, 
                policy_kwargs=custom_arch, 
                n_steps=int(2048*8/parallel_env), verbose=1,
                tensorboard_log="trained/tensorboard_logs/")
    # # Load model to continue previous training
    # model = PPO.load("trained/ckpts/PPO_0506_175240_forward_only/rl_model_240000_steps.zip", 
    #                   env, policy_kwargs=custom_arch, 
    #                   n_steps=int(2048*8/parallel_env), verbose=1, 
    #                   tensorboard_log="trained/tensorboard_logs/")

    model.learn(1e6, tb_log_name=exp_name, callback=checkpoint_callback)

    model.save("trained/" + exp_name)

