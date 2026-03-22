"""
Phase 4: RL Agent Training and Comparison
Compares different observation representations for navigation task:
1. Baseline: Raw pixels (CNN policy learns from scratch)
2. Encoder: Pretrained world encoder features (256-dim)
"""
import os
import numpy as np
import gymnasium as gym
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import json
import math

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_wrapper import GridWorldGymEnv


class ShapedRewardWrapper(gym.Wrapper):
    """Wrapper that adds distance-based reward shaping for faster learning."""
    
    def __init__(self, env, shaping_scale=0.5):
        super().__init__(env)
        self.prev_distance = None
        self.shaping_scale = shaping_scale
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        agent_pos = self.env.unwrapped.env.agent_pos
        target_pos = self.env.unwrapped.env.target_pos
        self.prev_distance = math.sqrt(
            (target_pos[0] - agent_pos[0])**2 + 
            (target_pos[1] - agent_pos[1])**2
        )
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        agent_pos = self.env.unwrapped.env.agent_pos
        target_pos = self.env.unwrapped.env.target_pos
        curr_distance = math.sqrt(
            (target_pos[0] - agent_pos[0])**2 + 
            (target_pos[1] - agent_pos[1])**2
        )
        
        # Shaped reward: bonus for getting closer
        if self.prev_distance is not None:
            distance_reward = (self.prev_distance - curr_distance) * self.shaping_scale
            reward += distance_reward
        
        self.prev_distance = curr_distance
        return obs, reward, terminated, truncated, info


class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards during training."""
    
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        
    def _on_step(self):
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    if 'infos' in self.locals and len(self.locals['infos']) > i:
                        info = self.locals['infos'][i]
                        if 'episode' in info:
                            ep_reward = info['episode']['r']
                            self.episode_rewards.append(ep_reward)
                            self.episode_lengths.append(info['episode']['l'])
                            # Success if reached goal (reward > threshold)
                            self.successes.append(1 if ep_reward > 0.5 else 0)
        return True
    
    def _on_training_end(self):
        with open(self.log_path, 'w') as f:
            json.dump({
                'rewards': self.episode_rewards,
                'lengths': self.episode_lengths,
                'successes': self.successes
            }, f)


def make_env(observation_mode, encoder_checkpoint=None, use_shaped_reward=True, 
             num_obstacles=2, max_steps=50):
    """Environment factory."""
    def _init():
        env = GridWorldGymEnv(
            observation_mode=observation_mode,
            encoder_checkpoint=encoder_checkpoint,
            grid_size=8,
            cell_size=8,
            num_obstacles=num_obstacles,
            max_steps=max_steps
        )
        if use_shaped_reward:
            env = ShapedRewardWrapper(env, shaping_scale=0.5)
        env = Monitor(env)
        return env
    return _init


def train_agent(observation_mode, encoder_checkpoint=None, total_timesteps=150000,
                save_name="agent", log_dir="logs", use_shaped_reward=True,
                num_obstacles=2, max_steps=50):
    """Train PPO agent."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training: {save_name}")
    print(f"  Mode: {observation_mode}")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  Obstacles: {num_obstacles}, Max steps: {max_steps}")
    print(f"{'='*60}\n")
    
    env = DummyVecEnv([make_env(observation_mode, encoder_checkpoint, use_shaped_reward,
                                num_obstacles, max_steps)])
    eval_env = DummyVecEnv([make_env(observation_mode, encoder_checkpoint, use_shaped_reward,
                                     num_obstacles, max_steps)])
    
    if observation_mode == 'pixels':
        policy = "CnnPolicy"
        policy_kwargs = None
    else:
        policy = "MlpPolicy"
        policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
    
    model = PPO(
        policy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    
    reward_logger = RewardLoggerCallback(
        log_path=os.path.join(log_dir, f"{save_name}_rewards.json")
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"checkpoints/{save_name}",
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=20
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_logger, eval_callback],
        progress_bar=False
    )
    
    model.save(f"checkpoints/{save_name}_final")
    
    # Training summary
    if reward_logger.successes:
        recent = reward_logger.successes[-100:] if len(reward_logger.successes) >= 100 else reward_logger.successes
        print(f"\n  Training Summary:")
        print(f"    Episodes: {len(reward_logger.episode_rewards)}")
        print(f"    Recent success rate: {100*sum(recent)/len(recent):.1f}%")
    
    print(f"  Model saved to: checkpoints/{save_name}_final")
    
    return model, reward_logger


def evaluate_agent(model, observation_mode, encoder_checkpoint=None, n_episodes=100,
                   num_obstacles=2, max_steps=50):
    """Evaluate agent."""
    env = GridWorldGymEnv(
        observation_mode=observation_mode,
        encoder_checkpoint=encoder_checkpoint,
        grid_size=8,
        cell_size=8,
        num_obstacles=num_obstacles,
        max_steps=max_steps
    )
    
    successes = 0
    total_rewards = []
    episode_lengths = []
    success_lengths = []
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if done or truncated:
                break
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        if total_reward > 0.5:
            successes += 1
            success_lengths.append(steps)
    
    return {
        'success_rate': successes / n_episodes * 100,
        'avg_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'avg_success_length': float(np.mean(success_lengths)) if success_lengths else 0
    }


def plot_results(baseline_logger, encoder_logger, save_path='logs/training_curves.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rewards
    ax1 = axes[0]
    window = 50
    
    if baseline_logger.episode_rewards:
        baseline_smooth = np.convolve(baseline_logger.episode_rewards,
                                      np.ones(window)/window, mode='valid')
        ax1.plot(baseline_smooth, label='Baseline (pixels)', color='orange', alpha=0.8)
    
    if encoder_logger.episode_rewards:
        encoder_smooth = np.convolve(encoder_logger.episode_rewards,
                                     np.ones(window)/window, mode='valid')
        ax1.plot(encoder_smooth, label='Encoder (ours)', color='blue', alpha=0.8)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (smoothed)')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Success rate
    ax2 = axes[1]
    window = 100
    
    if baseline_logger.successes:
        baseline_sr = np.convolve(baseline_logger.successes,
                                  np.ones(window)/window, mode='valid') * 100
        ax2.plot(baseline_sr, label='Baseline (pixels)', color='orange', alpha=0.8)
    
    if encoder_logger.successes:
        encoder_sr = np.convolve(encoder_logger.successes,
                                 np.ones(window)/window, mode='valid') * 100
        ax2.plot(encoder_sr, label='Encoder (ours)', color='blue', alpha=0.8)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Training Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    # Configuration - no obstacles so encoder direction prediction works
    NUM_OBSTACLES = 0  # No obstacles - encoder trained for direction, not obstacle avoidance
    MAX_STEPS = 50     # Standard episode length
    TIMESTEPS = 150000 # Sufficient for convergence
    
    print("="*60)
    print("PHASE 4: RL Agent Comparison")
    print("="*60)
    print(f"Grid: 8x8, Obstacles: {NUM_OBSTACLES}, Max steps: {MAX_STEPS}")
    print(f"Training timesteps: {TIMESTEPS}")
    print("="*60)
    
    # Train Baseline (raw pixels)
    baseline_model, baseline_logger = train_agent(
        observation_mode='pixels',
        total_timesteps=TIMESTEPS,
        save_name='baseline_agent',
        log_dir='logs/baseline',
        use_shaped_reward=True,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS
    )
    
    # Train Encoder-based agent (uses pretrained world encoder)
    encoder_model, encoder_logger = train_agent(
        observation_mode='encoder',
        encoder_checkpoint='checkpoints/phase2_world_encoder.pth',
        total_timesteps=TIMESTEPS,
        save_name='encoder_agent',
        log_dir='logs/encoder',
        use_shaped_reward=True,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS
    )
    
    # Plot training curves
    plot_results(baseline_logger, encoder_logger)
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION RESULTS (100 episodes each)")
    print("="*60)
    
    baseline_results = evaluate_agent(
        baseline_model,
        'pixels',
        n_episodes=100,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS
    )
    print(f"\n🟠 Baseline Agent (Raw Pixels):")
    print(f"   Success Rate: {baseline_results['success_rate']:.1f}%")
    print(f"   Avg Reward: {baseline_results['avg_reward']:.3f} ± {baseline_results['std_reward']:.3f}")
    print(f"   Avg Steps: {baseline_results['avg_length']:.1f}")
    
    encoder_results = evaluate_agent(
        encoder_model,
        'encoder',
        'checkpoints/phase2_world_encoder.pth',
        n_episodes=100,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS
    )
    print(f"\n🔵 Encoder Agent (Pretrained Features):")
    print(f"   Success Rate: {encoder_results['success_rate']:.1f}%")
    print(f"   Avg Reward: {encoder_results['avg_reward']:.3f} ± {encoder_results['std_reward']:.3f}")
    print(f"   Avg Steps: {encoder_results['avg_length']:.1f}")
    
    # Save results
    results = {
        'baseline': baseline_results,
        'encoder': encoder_results,
        'config': {
            'num_obstacles': NUM_OBSTACLES,
            'max_steps': MAX_STEPS,
            'timesteps': TIMESTEPS
        }
    }
    with open('logs/phase4_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to logs/phase4_results.json")
    print("="*60)


if __name__ == "__main__":
    main()
