# enhanced_test.py
from qkd_env import QKDRLEnv
import numpy as np

def test_environment():
    print("🧪 Testing Enhanced QKD Environment")
    print("=" * 50)
    
    env = QKDRLEnv()
    obs = env.reset()[0]
    print(f"Initial observation shape: {obs.shape}")
    
    # Test with different parameter combinations
    test_actions = [
        ([0.5, 0.05, 0.3, 0.5], "Balanced Parameters"),
        ([0.6, 0.08, 0.2, 0.3], "Conservative Setup"),
        ([0.4, 0.03, 0.4, 0.7], "High Filter Setup"),
        ([0.7, 0.1, 0.5, 0.1], "Aggressive Setup"),
    ]
    
    for action_vals, description in test_actions:
        print(f"\n--- Testing: {description} ---")
        action = np.array(action_vals, dtype=np.float32)
        
        env.reset()
        total_reward = 0
        key_rates = []
        qbers = []
        
        for i in range(20):  # Test 20 steps
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            key_rates.append(info['key'])
            qbers.append(info['qber'])
            
            if i % 5 == 0:  # Print every 5 steps
                print(f"  Step {i:2d}: R={reward:6.1f}, QBER={info['qber']:.4f}, Key={info['key']:.5f}, Done={done}")
            
            if done:
                print(f"  Episode ended at step {i}")
                break
        
        print(f"  Total Reward: {total_reward:.1f}")
        print(f"  Avg QBER: {np.mean(qbers):.4f}")
        print(f"  Avg Key Rate: {np.mean(key_rates):.5f}")
        print(f"  Max Key Rate: {np.max(key_rates):.5f}")
        
        # Check if we're generating any keys
        positive_keys = [k for k in key_rates if k > 0]
        if positive_keys:
            print(f"  ✅ Generated keys in {len(positive_keys)}/{len(key_rates)} steps!")
        else:
            print(f"  ❌ No key generation - QBER too high")
    
    print("\n" + "=" * 50)
    print("🎯 Environment test completed!")

if __name__ == "__main__":
    test_environment()