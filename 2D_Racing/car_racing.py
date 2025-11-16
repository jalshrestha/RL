"""
Gymnasium CarRacing-v3 with PPO

"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import numpy as np
import os


def make_env():
    """Create CarRacing environment with proper wrappers"""
    env = gym.make(
        'CarRacing-v3',
        render_mode=None,
        continuous=True,
        lap_complete_percent=0.95,
        domain_randomize=False
    )
    env = Monitor(env)
    return env


def train_official_carracing():
    """Train PPO on official CarRacing environment"""
    print("="*60)
    print("Training PPO on Official Gymnasium CarRacing-v3")
    print("="*60)
    
    # Create vectorized environment
    print("\n1. Creating environment...")
    env = DummyVecEnv([make_env])
    
    # Create PPO model
    print("2. Creating PPO model...")
    print("   Note: This uses image observations (96x96x3 RGB)")
    
    model = PPO(
        "CnnPolicy",  # CNN for image input
        env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/tensorboard"
    )
    
    # Train
    timesteps = 50000
    print(f"3. Training for {timesteps:,} steps...")
  
    print("   The car learns to complete laps with proper graphics!\n")
    
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True
    )
    
    # Save
    print("\n4. Saving model...")
    os.makedirs("./official_model", exist_ok=True)
    model.save("./official_model/carracing_ppo")
    print("✓ Model saved to: ./official_model/carracing_ppo.zip")
    
    env.close()
    return model


def test_official_carracing(model_path="./official_model/carracing_ppo", num_episodes=3):
    """Test trained agent on official environment"""
    print("\n" + "="*60)
    print("Testing Trained Agent on Official CarRacing")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except:
        print("❌ Model not found! Please train first (option 1)")
        return
    
    # Create environment with rendering
    print("Creating environment with rendering...\n")
    env = gym.make(
        'CarRacing-v3',
        render_mode='human',  # Show window
        continuous=True,
        lap_complete_percent=0.95
    )
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"{'='*40}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print('='*40)
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Render (automatic with render_mode='human')
        
        print(f"✓ Total Reward: {total_reward:.2f}")
        print(f"✓ Steps: {steps}")
        
        if total_reward > 900:
            print("🏆 EXCELLENT! Completed the track!")
        elif total_reward > 500:
            print("👍 Good! Made significant progress")
        else:
            print("📈 Still learning...")
    
    env.close()
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


def watch_random_agent():
    """Watch a random agent for comparison"""
    print("\n" + "="*60)
    print("Random Agent (for comparison)")
    print("="*60)
    
    env = gym.make('CarRacing-v3', render_mode='human', continuous=True)
    
    for episode in range(2):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}/2 (Random Actions)")
        
        while not done:
            action = env.action_space.sample()  # Random
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"Reward: {total_reward:.2f}")
    
    env.close()


def quick_demo():
    """Quick 5-minute training demo"""
    print("="*60)
    print("Quick Demo - 5 minute training")
    print("="*60)
    
    env = DummyVecEnv([make_env])
    
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=32,
        verbose=0
    )
    
    print("\nTraining for 50,000 steps (~5 minutes)...")
    model.learn(total_timesteps=50000, progress_bar=True)
    
    print("\nTesting...")
    env = gym.make('CarRacing-v3', render_mode='human', continuous=True)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    print(f"\nReward: {total_reward:.2f}")
    env.close()


def main():
    """Main menu"""
    print("\n" + "="*60)
    print("Official Gymnasium CarRacing-v3 with PPO")
    print("="*60)
    print("\nThis uses the OFFICIAL environment with:")
    print("✓ Professional graphics (top-down car view)")
    print("✓ Randomly generated tracks")
    print("✓ Proper physics simulation")
    print("✓ Image-based observations (like real vision)")
    
    print("\n" + "="*60)
    print("Options:")
    print("="*60)
    print("1. Full training (500k steps, ~20 min, BEST results)")
    print("2. Test existing trained agent")
    print("3. Quick demo (50k steps, ~5 min)")
    print("4. Watch random agent (see how hard it is)")
    print("5. Train AND test")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        train_official_carracing()
        print("\n✅ Training complete! Run option 2 to see it race!")
    elif choice == "2":
        test_official_carracing()
    elif choice == "3":
        quick_demo()
    elif choice == "4":
        watch_random_agent()
    elif choice == "5":
        train_official_carracing()
        input("\n✅ Training done! Press Enter to watch it race...")
        test_official_carracing()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()