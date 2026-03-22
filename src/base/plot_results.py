"""
Generate comparison plots for Phase 4 results.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison():
    # Load results
    with open('logs/comparison_results.json', 'r') as f:
        results = json.load(f)
    
    aligned = results['aligned']
    baseline = results['baseline']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Success Rate
    ax1 = axes[0]
    categories = ['Aligned\n(Our Method)', 'Baseline\n(CNN)']
    success_rates = [aligned['success_rate'], baseline['success_rate']]
    colors = ['#4CAF50', '#FF5722']
    bars1 = ax1.bar(categories, success_rates, color=colors)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate Comparison')
    ax1.set_ylim(0, max(success_rates) * 1.5 + 10)
    for bar, val in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=10)
    
    # Plot 2: Average Reward
    ax2 = axes[1]
    avg_rewards = [aligned['avg_reward'], baseline['avg_reward']]
    bars2 = ax2.bar(categories, avg_rewards, color=colors)
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Average Episode Reward')
    for bar, val in zip(bars2, avg_rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05, 
                f'{val:.3f}', ha='center', fontsize=10)
    
    # Plot 3: Average Steps
    ax3 = axes[2]
    avg_steps = [aligned['avg_length'], baseline['avg_length']]
    bars3 = ax3.bar(categories, avg_steps, color=colors)
    ax3.set_ylabel('Average Steps')
    ax3.set_title('Average Episode Length')
    ax3.set_ylim(0, 60)
    for bar, val in zip(bars3, avg_steps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('phase4_comparison.png', dpi=150)
    print("Saved phase4_comparison.png")
    
    # Print summary
    print("\n" + "="*50)
    print("PHASE 4 FINAL RESULTS")
    print("="*50)
    print(f"\nAligned Agent (Math-Informed):")
    print(f"  Success Rate: {aligned['success_rate']:.1f}%")
    print(f"  Avg Reward:   {aligned['avg_reward']:.3f}")
    print(f"  Avg Steps:    {aligned['avg_length']:.1f}")
    
    print(f"\nBaseline Agent (End-to-End CNN):")
    print(f"  Success Rate: {baseline['success_rate']:.1f}%")
    print(f"  Avg Reward:   {baseline['avg_reward']:.3f}")
    print(f"  Avg Steps:    {baseline['avg_length']:.1f}")
    
    improvement = aligned['success_rate'] - baseline['success_rate']
    print(f"\nImprovement: +{improvement:.1f}% success rate")

if __name__ == "__main__":
    plot_comparison()
