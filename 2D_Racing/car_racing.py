"""
CarRacing-v3 PPO Training Script
Jaljala Shrestha Lama - CSC 425 Final Project

PPO agent to race in Gymnasium's CarRacing environment.
Uses parallel envs, frame stacking, and reward shaping for better results.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
from datetime import datetime
import warnings

# ======================
# CONFIG
# ======================
CONFIG = {
    "n_envs": 8,
    "frame_stack": 4,
    "total_timesteps": 500_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "checkpoint_freq": 10_000,
    "eval_freq": 5_000,
    "n_eval_episodes": 5,
    "model_dir": "./official_model",
    "checkpoint_dir": "./checkpoints",
    "log_dir": "./logs/tensorboard",
}


def setup_device():
    """Pick best available device"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print("[+] Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"[+] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[+] Using CPU")
    return device


class RewardShapingWrapper(gym.Wrapper):
    """Modifies rewards - penalize grass, reward staying on track"""
    
    def __init__(self, env):
        super().__init__(env)
        self.consecutive_on_track = 0
        self.prev_speed = 0
        self.episode_reward = 0
        self.grass_penalty_count = 0
        
    def reset(self, **kwargs):
        self.consecutive_on_track = 0
        self.prev_speed = 0
        self.episode_reward = 0
        self.grass_penalty_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = reward
        
        # Check if on grass
        grass_region = obs[60:84, 40:56, :]
        green_ratio = np.mean(grass_region[:, :, 1]) / (np.mean(grass_region[:, :, 0]) + 1e-6)
        on_grass = green_ratio > 1.3
        
        if on_grass:
            shaped_reward -= 0.5
            self.consecutive_on_track = 0
            self.grass_penalty_count += 1
        else:
            self.consecutive_on_track += 1
            if self.consecutive_on_track > 20:
                shaped_reward += 0.1
        
        # Penalize wild steering
        steering = abs(action[0])
        if steering > 0.8:
            shaped_reward -= 0.05 * steering
        
        # Penalize hard braking at speed
        gas = action[1]
        brake = action[2]
        if brake > 0.5 and self.prev_speed > 50:
            shaped_reward -= 0.1
        
        # Bonus for accelerating on track
        if gas > 0.5 and not on_grass:
            shaped_reward += 0.05
        
        self.prev_speed = gas * 100
        self.episode_reward += shaped_reward
        
        info['shaped_reward'] = shaped_reward
        info['original_reward'] = reward
        info['on_grass'] = on_grass
        info['grass_penalties'] = self.grass_penalty_count
        
        return obs, shaped_reward, terminated, truncated, info


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert to grayscale - optional"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(96, 96, 1), dtype=np.uint8
        )
    
    def observation(self, obs):
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        return gray.astype(np.uint8)[..., np.newaxis]


class TrainingCallback(BaseCallback):
    """Logs progress and saves training curve"""
    
    def __init__(self, check_freq=1000, log_dir="./logs", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.mean_rewards = []
        self.timestamps = []
        self.best_mean_reward = -np.inf
        self.reward_window = deque(maxlen=100)
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        print("\n" + "="*60)
        print("Training started")
        print("="*60)
        print(f"Device: {self.model.device}")
        print(f"Total steps: {self.model._total_timesteps:,}")
        print(f"Envs: {self.model.n_envs}")
        print("="*60 + "\n")
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                    self.reward_window.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        if self.n_calls % self.check_freq == 0:
            self._log_progress()
        return True
    
    def _log_progress(self):
        elapsed = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / elapsed
        
        if len(self.reward_window) > 0:
            mean_reward = np.mean(self.reward_window)
            std_reward = np.std(self.reward_window)
            self.mean_rewards.append(mean_reward)
            self.timestamps.append(self.num_timesteps)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_marker = " ** new best"
            else:
                best_marker = ""
            
            progress = self.num_timesteps / self.model._total_timesteps
            bar_length = 30
            filled = int(bar_length * progress)
            bar = "#" * filled + "-" * (bar_length - filled)
            
            if progress > 0:
                eta_seconds = elapsed / progress - elapsed
                eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
            else:
                eta_str = "..."
            
            print(f"\n[{bar}] {progress*100:.1f}%")
            print(f"Steps: {self.num_timesteps:,} | Eps: {len(self.episode_rewards)}")
            print(f"Avg reward (100 ep): {mean_reward:.1f} +/- {std_reward:.1f}{best_marker}")
            print(f"Speed: {steps_per_sec:.0f} steps/s | ETA: {eta_str}")
    
    def _on_training_end(self):
        total_time = time.time() - self.start_time
        print("\n" + "="*60)
        print("Training done")
        print("="*60)
        print(f"Time: {total_time/60:.1f} min")
        print(f"Steps: {self.num_timesteps:,}")
        print(f"Episodes: {len(self.episode_rewards)}")
        print(f"Best avg: {self.best_mean_reward:.1f}")
        print(f"Avg speed: {self.num_timesteps/total_time:.0f} steps/s")
        print("="*60)
        self._save_training_curve()
    
    def _save_training_curve(self):
        if len(self.mean_rewards) < 2:
            return
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, alpha=0.3, label='Episode')
        if len(self.episode_rewards) > 10:
            smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            plt.plot(smoothed, label='Smoothed', linewidth=2)
        plt.axhline(y=900, color='g', linestyle='--', label='Target')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.timestamps, self.mean_rewards, 'b-', linewidth=2)
        plt.axhline(y=900, color='g', linestyle='--', label='Target')
        plt.xlabel('Steps')
        plt.ylabel('Avg Reward')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./Training_Curve.png', dpi=150)
        plt.close()
        print("Saved Training_Curve.png")


def make_env(rank, seed=0, use_reward_shaping=True):
    """Factory for creating envs"""
    def _init():
        env = gym.make(
            'CarRacing-v3',
            render_mode=None,
            continuous=True,
            lap_complete_percent=0.95,
            domain_randomize=False
        )
        if use_reward_shaping:
            env = RewardShapingWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_training_env(n_envs=8, use_reward_shaping=True):
    """Create parallel envs with frame stacking"""
    print(f"Creating {n_envs} envs...")
    
    env_fns = [make_env(i, use_reward_shaping=use_reward_shaping) for i in range(n_envs)]
    
    if n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method='fork')
        print("  Using SubprocVecEnv")
    else:
        env = DummyVecEnv(env_fns)
        print("  Using DummyVecEnv")
    
    env = VecFrameStack(env, n_stack=CONFIG["frame_stack"])
    print(f"  Frame stacking: {CONFIG['frame_stack']}")
    print(f"  Obs shape: {env.observation_space.shape}")
    
    return env


def linear_schedule(initial_value):
    """LR decays linearly"""
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule


def train_optimized(total_timesteps=None, n_envs=None, use_reward_shaping=True,
                    continue_training=False, model_path=None):
    """Main training function"""
    timesteps = total_timesteps or CONFIG["total_timesteps"]
    num_envs = n_envs or CONFIG["n_envs"]
    
    print("\n" + "="*60)
    print("PPO Training - CarRacing-v3")
    print("="*60)
    
    device = setup_device()
    
    if device == "mps":
        try:
            test_tensor = torch.zeros(1, device="mps")
            del test_tensor
        except Exception as e:
            print(f"MPS failed: {e}, using CPU")
            device = "cpu"
    
    print(f"\nEnv config:")
    print(f"  Parallel envs: {num_envs}")
    print(f"  Frame stack: {CONFIG['frame_stack']}")
    print(f"  Reward shaping: {use_reward_shaping}")
    
    env = create_training_env(n_envs=num_envs, use_reward_shaping=use_reward_shaping)
    eval_env = create_training_env(n_envs=1, use_reward_shaping=False)
    
    print(f"\nModel config:")
    print(f"  Policy: CnnPolicy")
    print(f"  LR: {CONFIG['learning_rate']}")
    print(f"  Batch: {CONFIG['batch_size']}")
    
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    
    if continue_training and model_path:
        print(f"\nLoading {model_path}...")
        model = PPO.load(model_path, env=env, device=device,
                         tensorboard_log=CONFIG["log_dir"])
        model.learning_rate = linear_schedule(CONFIG["learning_rate"] * 0.5)
    else:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=linear_schedule(CONFIG["learning_rate"]),
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            gamma=CONFIG["gamma"],
            gae_lambda=CONFIG["gae_lambda"],
            clip_range=CONFIG["clip_range"],
            ent_coef=CONFIG["ent_coef"],
            vf_coef=CONFIG["vf_coef"],
            max_grad_norm=CONFIG["max_grad_norm"],
            device=device,
            verbose=0,
            tensorboard_log=CONFIG["log_dir"],
            policy_kwargs={"normalize_images": True}
        )
    
    print(f"  Device: {model.device}")
    
    training_callback = TrainingCallback(check_freq=2000, log_dir=CONFIG["log_dir"])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CONFIG["checkpoint_freq"] // num_envs,
        save_path=CONFIG["checkpoint_dir"],
        name_prefix="carracing_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=CONFIG["model_dir"],
        log_path=CONFIG["log_dir"],
        eval_freq=CONFIG["eval_freq"] // num_envs,
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False
    )
    
    callbacks = [training_callback, checkpoint_callback, eval_callback]
    
    print(f"\nStarting {timesteps:,} steps...")
    print("-"*60)
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not continue_training
        )
    except KeyboardInterrupt:
        print("\n\nStopped early, saving...")
    
    final_path = f"{CONFIG['model_dir']}/carracing_ppo_final"
    model.save(final_path)
    print(f"\nSaved to: {final_path}.zip")
    
    env.close()
    eval_env.close()
    
    return model


def test_agent(model_path=None, num_episodes=5, deterministic=True):
    """Test the trained agent"""
    print("\n" + "="*60)
    print("Testing Agent")
    print("="*60)
    
    if model_path is None:
        best_path = f"{CONFIG['model_dir']}/best_model"
        final_path = f"{CONFIG['model_dir']}/carracing_ppo_final"
        
        if os.path.exists(f"{best_path}.zip"):
            model_path = best_path
            print("Loading best model")
        elif os.path.exists(f"{final_path}.zip"):
            model_path = final_path
            print("Loading final model")
        else:
            print("No model found!")
            return
    
    print(f"  Path: {model_path}")
    model = PPO.load(model_path)
    
    def make_test_env():
        return gym.make('CarRacing-v3', render_mode='human', continuous=True,
                        lap_complete_percent=0.95)
    
    test_env = DummyVecEnv([make_test_env])
    test_env = VecFrameStack(test_env, n_stack=CONFIG["frame_stack"])
    
    results = []
    
    for episode in range(num_episodes):
        obs = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nEp {episode + 1}/{num_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_array, info = test_env.step(action)
            done = done_array[0]
            total_reward += reward[0]
            steps += 1
        
        results.append(total_reward)
        print(f"  Reward: {total_reward:.1f}, Steps: {steps}")
    
    test_env.close()
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean: {np.mean(results):.1f} +/- {np.std(results):.1f}")
    print(f"Min: {np.min(results):.1f} | Max: {np.max(results):.1f}")
    print("="*60)


def quick_demo(timesteps=50_000):
    """Short training run to test setup"""
    print("\n" + "="*60)
    print("Quick Demo")
    print("="*60)
    print(f"Running {timesteps:,} steps\n")
    
    model = train_optimized(total_timesteps=timesteps, n_envs=4, use_reward_shaping=True)
    print("\nDone! Use test_agent() to see results")
    return model


def watch_random_agent(num_episodes=2):
    """Watch random agent for baseline"""
    print("\n" + "="*60)
    print("Random Agent Baseline")
    print("="*60)
    
    env = gym.make('CarRacing-v3', render_mode='human', continuous=True)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"Ep {episode + 1}/{num_episodes}")
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"  Reward: {total_reward:.1f}")
    
    env.close()
    print("\nRandom gets -50 to 100, trained should get 700-900+")


def benchmark_speed(timesteps=10_000):
    """Test training speed"""
    print("\n" + "="*60)
    print("Speed Benchmark")
    print("="*60)
    
    configs = [(1, "1 env"), (4, "4 envs"), (8, "8 envs")]
    results = []
    
    for n_envs, name in configs:
        print(f"\nTesting {name}...")
        env = create_training_env(n_envs=n_envs, use_reward_shaping=False)
        model = PPO("CnnPolicy", env, verbose=0, n_steps=256)
        
        start = time.time()
        model.learn(total_timesteps=timesteps, progress_bar=False)
        elapsed = time.time() - start
        
        speed = timesteps / elapsed
        results.append((name, speed))
        print(f"  {speed:.0f} steps/s")
        env.close()
    
    print("\n" + "="*60)
    print("Results:")
    for name, speed in results:
        print(f"  {name}: {speed:.0f} steps/s")
    print("="*60)


def main():
    """Menu"""
    print("\n" + "="*60)
    print("CarRacing PPO Trainer")
    print("="*60)
    print("\nSetup:")
    print("  - 8 parallel envs")
    print("  - 4-frame stacking")
    print("  - Reward shaping")
    print("  - Auto checkpointing")
    
    print("\n" + "="*60)
    print("Options:")
    print("="*60)
    print("1. Train (500k steps)")
    print("2. Test agent")
    print("3. Quick demo (50k)")
    print("4. Watch random agent")
    print("5. Train then test")
    print("6. Benchmark speed")
    print("7. Continue from checkpoint")
    
    choice = input("\nChoice (1-7): ").strip()
    
    if choice == "1":
        train_optimized()
    elif choice == "2":
        test_agent()
    elif choice == "3":
        quick_demo()
    elif choice == "4":
        watch_random_agent()
    elif choice == "5":
        train_optimized()
        input("\nDone! Press Enter to test...")
        test_agent()
    elif choice == "6":
        benchmark_speed()
    elif choice == "7":
        checkpoints = sorted([
            f for f in os.listdir(CONFIG["checkpoint_dir"]) 
            if f.endswith('.zip')
        ]) if os.path.exists(CONFIG["checkpoint_dir"]) else []
        
        if checkpoints:
            latest = checkpoints[-1]
            path = f"{CONFIG['checkpoint_dir']}/{latest[:-4]}"
            print(f"Found: {latest}")
            train_optimized(continue_training=True, model_path=path)
        else:
            print("No checkpoints, starting fresh...")
            train_optimized()
    else:
        print("Invalid!")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
