"""
Research-Grade QKD RL Environment for ICLR/ICML/AAAI Submission
Enhanced with sophisticated observation space design and robust episode management
Based on recent advances in Quantum Reinforcement Learning research
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, Tuple, Any, Optional, List
from qkd_simulator import simulate_qkd, reset_simulation
import logging

class QKDRLEnvironment(gym.Env):
    """
    Advanced Quantum Key Distribution Reinforcement Learning Environment
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 history_length: int = 12,
                 max_episode_steps: int = 8000,
                 observation_noise: float = 0.008,
                 reward_scaling: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize Advanced QKD RL Environment
        """
        super().__init__()
        
        self.history_length = history_length
        self.max_episode_steps = max_episode_steps
        self.observation_noise = observation_noise
        self.reward_scaling = reward_scaling
        
        if seed is not None:
            self.seed(seed)

        # --- Reward shaping knobs ---
        # Compress huge raw simulator rewards so shaping signal can matter
        self.reward_base_weight = 1e-3   # multiply raw reward by this

        # Weights and thresholds for shaping terms
        self.reward_shaping = {
            "qber_baseline": 0.50,          # where your episodes currently sit (~0.5)
            "qber_target": 0.11,            # secure threshold for BB84-like regimes
            "qber_weight": 2_000.0,         # reward for lowering QBER below baseline
            "keyrate_weight": 50_000.0,     # reward for >0 key rate
            "survival_bonus": 5.0,          # small per-step bonus
            "violation_penalty": -10_000.0, # penalty if security_violation True
            "milestone_bonus": 100_000.0    # big bonus when qber < target & key_rate > 0
        }
        
        # Enhanced action space
        self.action_space = spaces.Box(
            low=np.array([0.35, 0.01, 0.05, 0.0], dtype=np.float32),
            high=np.array([0.85, 0.15, 0.8, 1.0], dtype=np.float32),
            dtype=np.float32,
            seed=seed
        )
        
        # Enhanced observation space: current + historical
        obs_dim = 10
        total_obs_dim = obs_dim * (history_length + 1)
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0, 
            shape=(total_obs_dim,),
            dtype=np.float32,
            seed=seed
        )
        
        # State tracking
        self.history = deque(maxlen=history_length)
        self.step_count = 0
        self.episode_count = 0
        
        # Enhanced episode statistics for research
        self.episode_stats = {
            "rewards": [],
            "qbers": [], 
            "key_rates": [],
            "actions": [],
            "attack_intensities": [],
            "attack_strategies": [],
            "filter_effectiveness": [],
            "episode_return": 0.0,
            "security_violations": 0,
            "max_key_rate": 0.0,
            "mean_qber": 0.0,
            "qber_stability": 0.0,
            "convergence_metrics": []
        }
        
        # Research metrics tracking
        self.performance_history = {
            "episode_returns": [],
            "episode_lengths": [],
            "final_key_rates": [],
            "security_violation_counts": []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        return [seed]
    
    def _construct_observation(self, current_obs: np.ndarray) -> np.ndarray:
        """
        Construct full observation including temporal history
        Research note: Temporal information crucial for attack pattern detection
        """
        # Add realistic sensor noise (Gaussian)
        if self.observation_noise > 0:
            noise = np.random.normal(0, self.observation_noise, current_obs.shape)
            current_obs = current_obs + noise
            current_obs = np.clip(current_obs, 0, 1)  # Maintain valid range
        
        # Combine current observation with history
        full_obs = [current_obs]
        full_obs.extend(list(self.history))
        
        # Pad with zeros if history not full
        while len(full_obs) < self.history_length + 1:
            full_obs.append(np.zeros_like(current_obs))
        
        return np.concatenate(full_obs).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step with enhanced tracking"""
        self.step_count += 1
        
        # Ensure action is properly shaped and bounded
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Convert action to parameter dictionary
        params = {
            "basis_bias": float(action[0]),
            "error_threshold": float(action[1]), 
            "privacy_strength": float(action[2]),
            "filter_aggressiveness": float(action[3])
        }

        # Run simulation
        obs, reward, done = simulate_qkd(params, self.step_count)

        # -------- Reward Shaping --------
        # Extract metrics from obs (works if obs is dict; falls back safely otherwise)
        if isinstance(obs, dict):
            qber = float(obs.get("qber", 0.5))
            key_rate = float(obs.get("key_rate", 0.0))
            security_violation = bool(obs.get("security_violation", False) or obs.get("violation", False))
        else:
            # Fallbacks if obs is array-like; adjust indices if you have them
            qber = 0.5
            key_rate = 0.0
            security_violation = False

        # Compress the raw simulator reward so shaping can influence learning
        base = reward * self.reward_base_weight

        # Progress rewards
        qber_progress = (self.reward_shaping["qber_baseline"] - qber) * self.reward_shaping["qber_weight"]
        keyrate_gain  = max(0.0, key_rate) * self.reward_shaping["keyrate_weight"]
        survival      = self.reward_shaping["survival_bonus"]

        # Penalties / bonuses
        violation_term = self.reward_shaping["violation_penalty"] if security_violation else 0.0
        milestone = (
            self.reward_shaping["milestone_bonus"]
            if (qber < self.reward_shaping["qber_target"] and key_rate > 0.0)
            else 0.0
        )

        shaped_reward = base + qber_progress + keyrate_gain + survival + violation_term + milestone

        # Clip for stability (keeps gradients sane)
        shaped_reward = float(np.clip(shaped_reward, -1e6, 1e6))

        # Final scaling (keeps backward-compat with your existing knob)
        scaled_reward = shaped_reward * self.reward_scaling
        # --------------------------------
        
        # Update episode statistics
        self._update_episode_stats(obs, scaled_reward, action, params)
        
        # Check for maximum episode length
        if self.step_count >= self.max_episode_steps:
            done = True
            
        # Construct full observation with history
        full_obs = self._construct_observation(obs)
        self.history.append(obs.copy())
        
        # Comprehensive info dictionary for research analysis
        info = self._construct_info_dict(obs, params, done, scaled_reward)

        # Add reward breakdown for diagnostics
        info.setdefault("reward_components", {})
        info["reward_components"].update({
            "raw_reward": float(reward),
            "base": float(base),
            "qber_progress": float(qber_progress),
            "keyrate_gain": float(keyrate_gain),
            "survival": float(survival),
            "violation": float(violation_term),
            "milestone": float(milestone),
            "shaped_reward": float(shaped_reward),
            "qber": float(qber),
            "key_rate": float(key_rate),
            "security_violation": bool(security_violation),
        })

        
        return full_obs, scaled_reward, done, False, info
    
    def _update_episode_stats(self, obs: np.ndarray, reward: float, 
                            action: np.ndarray, params: Dict[str, float]):
        """Update comprehensive episode statistics"""
        self.episode_stats["rewards"].append(reward)
        self.episode_stats["qbers"].append(float(obs[0]))
        self.episode_stats["key_rates"].append(float(obs[1]))
        self.episode_stats["actions"].append(action.copy())
        self.episode_stats["attack_intensities"].append(float(obs[5]))
        self.episode_stats["attack_strategies"].append(int(obs[4] * 7))  # Denormalized
        self.episode_stats["filter_effectiveness"].append(float(obs[6]))
        self.episode_stats["episode_return"] += reward
        
        # Track maximum key rate achieved
        if obs[1] > self.episode_stats["max_key_rate"]:
            self.episode_stats["max_key_rate"] = float(obs[1])
        
        # Track security violations
        if obs[7] > 0.5:  # Threshold violation flag
            self.episode_stats["security_violations"] += 1
    
    def _construct_info_dict(self, obs: np.ndarray, params: Dict[str, float], 
                           done: bool, reward: float) -> Dict[str, Any]:
        """Construct comprehensive info dictionary for research"""
        info = {
            # Current state information
            "qber": float(obs[0]),
            "key_rate": float(obs[1]), 
            "phase_drift": float((obs[2] - 0.5) * 0.025),  # Denormalized
            "dark_drift": float((obs[3] - 0.5) * 1e-6),    # Denormalized
            "attack_strategy": int(obs[4] * 7),             # Denormalized to strategy ID
            "attack_intensity": float(obs[5]),
            "filter_setting": float(obs[6]),
            "threshold_violation": bool(obs[7] > 0.5),
            "temperature_drift": float((obs[8] - 0.5) * 0.02) if len(obs) > 8 else 0.0,
            "attack_sophistication": float(obs[9]) if len(obs) > 9 else 0.0,
            
            # Control parameters
            "basis_bias": params["basis_bias"],
            "error_threshold": params["error_threshold"],
            "privacy_strength": params["privacy_strength"],
            "filter_aggressiveness": params["filter_aggressiveness"],
            
            # Episode metadata
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "current_reward": reward,
            
            # Performance tracking (required for stable-baselines3 Monitor)
            "key_rate_history": self.episode_stats["key_rates"][-100:] if len(self.episode_stats["key_rates"]) >= 100 else self.episode_stats["key_rates"],
            "qber_history": self.episode_stats["qbers"][-100:] if len(self.episode_stats["qbers"]) >= 100 else self.episode_stats["qbers"]
        }
        
        # Add episode summary if done
        if done:
            # Calculate episode statistics
            episode_qbers = np.array(self.episode_stats["qbers"])
            episode_keys = np.array(self.episode_stats["key_rates"])
            
            # Calculate stability metrics
            qber_stability = np.std(episode_qbers) if len(episode_qbers) > 1 else 0.0
            key_consistency = np.std(episode_keys) if len(episode_keys) > 1 else 0.0
            
            episode_summary = {
                "r": self.episode_stats["episode_return"],
                "l": self.step_count,
                "mean_qber": float(np.mean(episode_qbers)) if len(episode_qbers) > 0 else 0.0,
                "mean_key_rate": float(np.mean(episode_keys)) if len(episode_keys) > 0 else 0.0,
                "max_key_rate": self.episode_stats["max_key_rate"],
                "security_violations": self.episode_stats["security_violations"],
                "violation_rate": self.episode_stats["security_violations"] / max(self.step_count, 1),
                "qber_stability": float(qber_stability),
                "key_consistency": float(key_consistency),
                "final_qber": float(episode_qbers[-1]) if len(episode_qbers) > 0 else 0.0,
                "final_key_rate": float(episode_keys[-1]) if len(episode_keys) > 0 else 0.0,
                
                # Additional research metrics
                "attack_diversity": len(set(self.episode_stats["attack_strategies"])),
                "mean_attack_intensity": float(np.mean(self.episode_stats["attack_intensities"])) if self.episode_stats["attack_intensities"] else 0.0,
                "filter_utilization": float(np.mean(self.episode_stats["filter_effectiveness"])) if self.episode_stats["filter_effectiveness"] else 0.0,
                
                # Performance classification for analysis
                "performance_class": self._classify_performance(self.episode_stats["max_key_rate"], episode_summary["mean_qber"] if "mean_qber" in locals() else 0.0),
                
                # Data for plotting (last 100 points for memory efficiency)
                "qber_trajectory": episode_qbers[-100:].tolist() if len(episode_qbers) > 0 else [],
                "key_trajectory": episode_keys[-100:].tolist() if len(episode_keys) > 0 else []
            }
            
            info["episode"] = episode_summary
            
            # Update performance history for long-term tracking
            self.performance_history["episode_returns"].append(episode_summary["r"])
            self.performance_history["episode_lengths"].append(episode_summary["l"])
            self.performance_history["final_key_rates"].append(episode_summary["final_key_rate"])
            self.performance_history["security_violation_counts"].append(episode_summary["security_violations"])
            
            # Log episode summary
            self.logger.info(f"Episode {self.episode_count} completed: "
               f"Return={episode_summary['r']:.1f}, "
               f"Length={episode_summary['l']}, "
               f"Mean QBER={episode_summary['mean_qber']:.6f}, "
               f"Mean Key Rate={episode_summary['mean_key_rate']:.6f}, "
               f"Max Key Rate={episode_summary['max_key_rate']:.6f}, "
               f"Class={episode_summary['performance_class']}")
        
        return info
    
    def _classify_performance(self, max_key_rate: float, mean_qber: float) -> str:
        """Classify episode performance for research analysis"""
        if max_key_rate >= 0.25 and mean_qber <= 0.05:
            return "Excellent"
        elif max_key_rate >= 0.15 and mean_qber <= 0.08:
            return "Good"
        elif max_key_rate >= 0.08 and mean_qber <= 0.12:
            return "Fair"
        elif max_key_rate >= 0.02 and mean_qber <= 0.18:
            return "Poor"
        else:
            return "Failed"
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode with enhanced initialization"""
        if seed is not None:
            self.seed(seed)
            
        # Reset simulation state
        reset_simulation()
        
        # Reset episode state
        self.step_count = 0
        self.episode_count += 1
        self.history.clear()
        
        # Reset episode statistics
        self.episode_stats = {
            "rewards": [],
            "qbers": [],
            "key_rates": [],
            "actions": [], 
            "attack_intensities": [],
            "attack_strategies": [],
            "filter_effectiveness": [],
            "episode_return": 0.0,
            "security_violations": 0,
            "max_key_rate": 0.0,
            "mean_qber": 0.0,
            "qber_stability": 0.0,
            "convergence_metrics": []
        }
        
        # Get initial observation with reasonable defaults
        dummy_params = {
            "basis_bias": 0.5,
            "error_threshold": 0.08,
            "privacy_strength": 0.25,
            "filter_aggressiveness": 0.3
        }
        
        initial_obs, _, _ = simulate_qkd(dummy_params, 0)
        
        # Initialize history with zeros (no prior information)
        zero_obs = np.zeros_like(initial_obs)
        for _ in range(self.history_length):
            self.history.append(zero_obs)
        
        # Construct full observation
        full_obs = self._construct_observation(initial_obs)
        
        # Reset info
        info = {
            "episode_count": self.episode_count,
            "max_episode_steps": self.max_episode_steps,
            "history_length": self.history_length
        }
        
        return full_obs, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment state for visualization/debugging"""
        if mode == "human" and hasattr(self, 'episode_stats') and self.episode_stats["qbers"]:
            recent_qber = self.episode_stats["qbers"][-1]
            recent_key = self.episode_stats["key_rates"][-1]
            recent_attack = self.episode_stats["attack_intensities"][-1] if self.episode_stats["attack_intensities"] else 0.0
            
            print(f"Step {self.step_count}: QBER={recent_qber:.4f}, "
                  f"Key Rate={recent_key:.4f}, Attack={recent_attack:.3f}")
        
        elif mode == "rgb_array":
            # Create a simple visualization for video recording
            import matplotlib.pyplot as plt
            from io import BytesIO
            
            if self.episode_stats["qbers"] and len(self.episode_stats["qbers"]) > 1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                
                steps = range(len(self.episode_stats["qbers"]))
                
                ax1.plot(steps, self.episode_stats["qbers"], 'r-', alpha=0.7, label='QBER')
                ax1.set_ylabel('QBER')
                ax1.set_title(f'QKD Performance - Episode {self.episode_count}')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                ax2.plot(steps, self.episode_stats["key_rates"], 'g-', alpha=0.7, label='Key Rate')
                ax2.set_ylabel('Key Rate')
                ax2.set_xlabel('Steps')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                
                # Convert to RGB array
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                plt.close()
                
                # This would require PIL/Pillow to properly convert
                # For now, return None to avoid dependencies
                return None
        
        return None
    
    def close(self):
        """Clean up environment resources"""
        pass
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get comprehensive episode statistics for research analysis"""
        if not self.performance_history["episode_returns"]:
            return {}
        
        recent_episodes = min(100, len(self.performance_history["episode_returns"]))
        recent_returns = self.performance_history["episode_returns"][-recent_episodes:]
        recent_key_rates = self.performance_history["final_key_rates"][-recent_episodes:]
        recent_violations = self.performance_history["security_violation_counts"][-recent_episodes:]
        
        stats = {
            "total_episodes": self.episode_count,
            "recent_episodes_analyzed": recent_episodes,
            "mean_return": float(np.mean(recent_returns)),
            "std_return": float(np.std(recent_returns)),
            "mean_final_key_rate": float(np.mean(recent_key_rates)),
            "max_key_rate_achieved": float(np.max(recent_key_rates)),
            "mean_security_violations": float(np.mean(recent_violations)),
            "success_rate": float(np.mean([rate > 0.05 for rate in recent_key_rates])),  # 5% threshold
            "episode_length_mean": float(np.mean(self.performance_history["episode_lengths"][-recent_episodes:])),
            
            # Performance distribution
            "excellent_episodes": sum(1 for rate in recent_key_rates if rate >= 0.25),
            "good_episodes": sum(1 for rate in recent_key_rates if 0.15 <= rate < 0.25),
            "fair_episodes": sum(1 for rate in recent_key_rates if 0.08 <= rate < 0.15),
            "poor_episodes": sum(1 for rate in recent_key_rates if 0.02 <= rate < 0.08),
            "failed_episodes": sum(1 for rate in recent_key_rates if rate < 0.02),
        }
        
        return stats

# Factory functions for easy environment creation
def make_qkd_env(**kwargs):
    """Factory function for creating standard QKD environment"""
    return QKDRLEnvironment(**kwargs)

def make_qkd_research_env(difficulty="medium", **kwargs):
    """Factory function for creating research-configured QKD environment"""
    configs = {
        "easy": {
            "history_length": 8,
            "max_episode_steps": 5000,
            "observation_noise": 0.005,
            "reward_scaling": 1.0
        },
        "medium": {
            "history_length": 12,
            "max_episode_steps": 8000,
            "observation_noise": 0.008,
            "reward_scaling": 1.0
        },
        "hard": {
            "history_length": 16,
            "max_episode_steps": 12000,
            "observation_noise": 0.012,
            "reward_scaling": 0.5
        }
    }
    
    config = configs.get(difficulty, configs["medium"])
    config.update(kwargs)  # Override with user-provided kwargs
    
    return QKDRLEnvironment(**config)