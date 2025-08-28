"""
Research-Grade QKD Simulator with Adversarial Drift
For submission to ICLR/ICML/AAAI conferences

Key Features:
- Realistic BB84 protocol implementation with quantum noise modeling
- Multi-modal adversarial attacks with temporal evolution
- Hardware drift modeling with correlated Gaussian processes
- Curriculum learning compatible environment
- Metrics aligned with quantum information theory standards
- Enhanced reward engineering based on recent QRL literature
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Set random seed for reproducibility
rng = np.random.default_rng(42)

class AttackStrategy(Enum):
    PASSIVE = 0
    INTERCEPT_RESEND = 1
    UNAMBIGUOUS_STATE_DISCRIMINATION = 2
    TIME_SHIFT_ATTACK = 3
    PHOTON_NUMBER_SPLITTING = 4
    BLINDING_ATTACK = 5
    TROJAN_HORSE_ATTACK = 6  # Advanced attack for research

@dataclass
class QKDSystemParameters:
    """System parameters for realistic QKD simulation based on commercial systems"""
    base_rate: float = 2.5e6  # Hz - increased for modern systems
    fiber_loss_db_km: float = 0.15  # dB/km
    distance_km: float = 30  # km - longer distance for realistic scenario
    detector_efficiency: float = 0.18  # Modern detector efficiency
    dark_count_rate: float = 5e-6  # Hz
    error_correction_efficiency: float = 1.12  # Realistic EC overhead
    privacy_amplification_overhead: float = 1.05  # PA overhead
    
    def __post_init__(self):
        # Calculate channel transmittance
        self.eta_channel = 10 ** (-self.fiber_loss_db_km * self.distance_km / 10)
        self.eta_total = self.eta_channel * self.detector_efficiency

@dataclass 
class AdversarialState:
    """Tracks evolving adversarial conditions with enhanced sophistication"""
    strategy: AttackStrategy = AttackStrategy.PASSIVE
    intensity: float = 0.0
    last_switch_step: int = 0
    evolution_rate: float = 0.1
    persistence_factor: float = 0.85  # How long attacks persist
    sophistication_level: int = 1  # Increases over time

class HardwareDriftModel:
    """Advanced Markov model for correlated hardware parameter drift"""
    def __init__(self):
        self.phase_drift = 0.0
        self.dark_rate_drift = 0.0
        self.temperature_drift = 0.0
        self.polarization_drift = 0.0
        self.reset()
    
    def reset(self):
        self.phase_drift = 0.0
        self.dark_rate_drift = 0.0
        self.temperature_drift = 0.0
        self.polarization_drift = 0.0
    
    def evolve(self) -> Tuple[float, float, float, float]:
        # Correlated Ornstein-Uhlenbeck processes for realistic drift
        # Phase drift (most critical for QKD)
        self.phase_drift = 0.96 * self.phase_drift + rng.normal(0, 0.0004)
        self.phase_drift = np.clip(self.phase_drift, -0.025, 0.025)
        
        # Dark count rate drift (temperature dependent)
        temp_correlation = 0.3 * self.temperature_drift
        self.dark_rate_drift = 0.98 * self.dark_rate_drift + temp_correlation + rng.normal(0, 1e-7)
        self.dark_rate_drift = np.clip(self.dark_rate_drift, -3e-6, 3e-6)
        
        # Temperature drift (slowly varying)
        self.temperature_drift = 0.995 * self.temperature_drift + rng.normal(0, 0.001)
        self.temperature_drift = np.clip(self.temperature_drift, -0.02, 0.02)
        
        # Polarization drift (fast varying)
        self.polarization_drift = 0.9 * self.polarization_drift + rng.normal(0, 0.002)
        self.polarization_drift = np.clip(self.polarization_drift, -0.03, 0.03)
        
        return self.phase_drift, self.dark_rate_drift, self.temperature_drift, self.polarization_drift

class CurriculumManager:
    """Enhanced curriculum learning progression based on recent QRL research"""
    def __init__(self):
        # Research-grade curriculum with smooth progression
        self.difficulty_phases = [
            {"steps": 0, "max_intensity": 0.0, "strategies": [AttackStrategy.PASSIVE], "noise_factor": 0.8},
            {"steps": 25000, "max_intensity": 0.08, "strategies": [AttackStrategy.PASSIVE, AttackStrategy.INTERCEPT_RESEND], "noise_factor": 0.9}, 
            {"steps": 75000, "max_intensity": 0.18, "strategies": [AttackStrategy.PASSIVE, AttackStrategy.INTERCEPT_RESEND, AttackStrategy.UNAMBIGUOUS_STATE_DISCRIMINATION], "noise_factor": 1.0},
            {"steps": 150000, "max_intensity": 0.3, "strategies": [AttackStrategy.PASSIVE, AttackStrategy.INTERCEPT_RESEND, AttackStrategy.UNAMBIGUOUS_STATE_DISCRIMINATION, AttackStrategy.TIME_SHIFT_ATTACK], "noise_factor": 1.1},
            {"steps": 250000, "max_intensity": 0.45, "strategies": list(AttackStrategy)[:6], "noise_factor": 1.2},
            {"steps": 400000, "max_intensity": 0.6, "strategies": list(AttackStrategy), "noise_factor": 1.25},
        ]
    
    def get_phase(self, step: int) -> Dict:
        for i in reversed(range(len(self.difficulty_phases))):
            if step >= self.difficulty_phases[i]["steps"]:
                return self.difficulty_phases[i]
        return self.difficulty_phases[0]

# Global state objects
system_params = QKDSystemParameters()
adversary = AdversarialState()
hardware = HardwareDriftModel()
curriculum = CurriculumManager()

def update_adversarial_state(step: int) -> None:
    """Update adversarial strategy using sophisticated curriculum learning"""
    phase = curriculum.get_phase(step)
    
    # Dynamic switching probability based on training progress
    if step < 10000:
        switch_prob = 0.0003  # Very rare changes during early learning
    elif step < 100000:
        switch_prob = 0.002 * (step / 100000)  # Gradual increase
    elif step < 500000:
        switch_prob = 0.008   # Moderate adaptation
    else:
        switch_prob = 0.015   # Full adaptation mode
    
    # Add persistence to attacks (realistic attacker behavior)
    steps_since_switch = step - adversary.last_switch_step
    persistence_decay = np.exp(-steps_since_switch / 5000) * adversary.persistence_factor
    
    # Switch strategy with persistence consideration
    if rng.random() < switch_prob * (1 - persistence_decay):
        adversary.strategy = rng.choice(phase["strategies"])
        # Intensity varies based on attack sophistication
        base_intensity = rng.uniform(0, phase["max_intensity"])
        sophistication_bonus = min(0.15, step / 1000000)  # Increases over time
        adversary.intensity = min(0.8, base_intensity + sophistication_bonus)
        adversary.last_switch_step = step
        adversary.sophistication_level = min(5, step // 100000 + 1)

def binary_entropy(p: float) -> float:
    """Shannon binary entropy function with numerical stability"""
    if p <= 1e-15 or p >= (1 - 1e-15):
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def calculate_attack_qber(strategy: AttackStrategy, intensity: float, sophistication: int) -> float:
    """Calculate QBER contribution from specific attack strategy with sophistication"""
    base_attack_qber = {
        AttackStrategy.PASSIVE: 0.0,
        AttackStrategy.INTERCEPT_RESEND: 0.25,
        AttackStrategy.UNAMBIGUOUS_STATE_DISCRIMINATION: 0.09,
        AttackStrategy.TIME_SHIFT_ATTACK: 0.04,
        AttackStrategy.PHOTON_NUMBER_SPLITTING: 0.13,
        AttackStrategy.BLINDING_ATTACK: 0.16 + 0.06 * rng.random(),
        AttackStrategy.TROJAN_HORSE_ATTACK: 0.22 + 0.08 * rng.random()
    }
    
    # Sophistication reduces detectability but maintains QBER impact
    base_qber = base_attack_qber[strategy]
    sophistication_factor = 1.0 + 0.1 * (sophistication - 1)  # More sophisticated attacks
    
    return intensity * base_qber * sophistication_factor

def simulate_qkd(params: Dict[str, float], step: int) -> Tuple[np.ndarray, float, bool]:
    """
    Main QKD simulation function with enhanced realism
    
    Args:
        params: Dictionary with keys: basis_bias, error_threshold, privacy_strength, filter_aggressiveness
        step: Current simulation step
    
    Returns:
        Tuple of (observations, reward, done)
    """
    # Update dynamic components
    update_adversarial_state(step)
    phase_drift, dark_drift, temp_drift, pol_drift = hardware.evolve()
    
    # Extract control parameters with bounds checking
    basis_bias = np.clip(params["basis_bias"], 0.35, 0.85)
    error_threshold = np.clip(params["error_threshold"], 0.01, 0.15)
    privacy_strength = np.clip(params["privacy_strength"], 0.05, 0.8)
    filter_strength = np.clip(params["filter_aggressiveness"], 0.0, 1.0)
    
    # Calculate channel parameters with correlated drift effects
    phase_impact = 1 - 0.4 * abs(phase_drift)
    pol_impact = 1 - 0.25 * abs(pol_drift)
    temp_impact = 1 - 0.15 * abs(temp_drift)
    
    eta_effective = system_params.eta_total * phase_impact * pol_impact * temp_impact
    eta_effective = max(0, eta_effective)  # Ensure non-negative
    
    dark_count_effective = system_params.dark_count_rate + abs(dark_drift)
    
    # Channel QBER with realistic quantum noise modeling
    if eta_effective > 1e-10:
        # Quantum channel noise
        quantum_noise = (1 - eta_effective) / 2
        # Dark count contribution
        dark_contribution = dark_count_effective / max(eta_effective, 1e-10)
        # Basis mismatch contribution (realistic modeling)
        basis_mismatch = 0.5 * (1 - 2 * min(basis_bias, 1 - basis_bias))
        qber_channel = quantum_noise + dark_contribution + 0.3 * basis_mismatch
    else:
        qber_channel = 0.5
    
    # Attack QBER with adaptive filtering
    qber_attack_raw = calculate_attack_qber(
        adversary.strategy, 
        adversary.intensity, 
        adversary.sophistication_level
    )
    
    # Filter effectiveness depends on attack type and filter strength
    filter_effectiveness = {
        AttackStrategy.PASSIVE: 0.0,
        AttackStrategy.INTERCEPT_RESEND: 0.7,
        AttackStrategy.UNAMBIGUOUS_STATE_DISCRIMINATION: 0.4,
        AttackStrategy.TIME_SHIFT_ATTACK: 0.3,
        AttackStrategy.PHOTON_NUMBER_SPLITTING: 0.6,
        AttackStrategy.BLINDING_ATTACK: 0.5,
        AttackStrategy.TROJAN_HORSE_ATTACK: 0.2  # Hard to filter
    }
    
    filter_reduction = filter_strength * filter_effectiveness[adversary.strategy]
    qber_attack = qber_attack_raw * (1 - filter_reduction)
    
    # Environmental noise (temperature, vibration, etc.)
    phase = curriculum.get_phase(step)
    noise_factor = phase.get("noise_factor", 1.0)
    qber_environment = (0.002 + 0.003 * rng.random()) * noise_factor
    
    # Total QBER with realistic bounds
    total_qber = np.clip(qber_channel + qber_attack + qber_environment, 0.0, 0.5)
    
    # Enhanced key rate calculation following quantum information theory
    raw_rate = system_params.base_rate * eta_effective
    
    # Sifted key rate with basis selection efficiency
    basis_efficiency = 2 * basis_bias * (1 - basis_bias)  # Maximized at 0.5
    sifted_rate = raw_rate * basis_efficiency
    
    if sifted_rate > 1e-6 and total_qber < 0.5:
        # Information reconciliation with realistic efficiency
        if total_qber > 0.5:
            leak_error_correction = sifted_rate  # No key can be extracted
        else:
            h_qber = binary_entropy(total_qber)
            leak_error_correction = system_params.error_correction_efficiency * sifted_rate * h_qber
        
        # Privacy amplification with enhanced security
        leak_privacy_amp = (system_params.privacy_amplification_overhead * 
                          sifted_rate * privacy_strength * h_qber)
        
        # Secure key rate
        secure_bits = max(0, sifted_rate - leak_error_correction - leak_privacy_amp)
        key_rate_normalized = secure_bits / max(sifted_rate, 1e-10) if sifted_rate > 1e-10 else 0.0
    else:
        key_rate_normalized = 0.0
    
    # Ensure key_rate_normalized is properly bounded
    key_rate_normalized = np.clip(key_rate_normalized, 0.0, 1.0)
    
    # Enhanced observation vector (10-dimensional for richer state representation)
    observations = np.array([
        total_qber,                                         # 0: Current QBER
        key_rate_normalized,                               # 1: Normalized key rate
        (phase_drift * 40 + 0.5),                         # 2: Phase drift (scaled)
        (dark_drift * 1e6 + 0.5),                         # 3: Dark count drift (scaled)
        adversary.strategy.value / len(AttackStrategy),     # 4: Attack strategy (normalized)
        adversary.intensity,                               # 5: Attack intensity
        filter_strength,                                   # 6: Filter setting
        float(total_qber > error_threshold),               # 7: Threshold violation flag
        (temp_drift * 25 + 0.5),                          # 8: Temperature drift (scaled)
        adversary.sophistication_level / 5.0               # 9: Attack sophistication (normalized)
    ], dtype=np.float32)
    
    # Ensure all observations are properly bounded [0, 1]
    observations = np.clip(observations, 0.0, 1.0)
    
    # Research-grade reward function
    reward = compute_advanced_reward(
        key_rate=key_rate_normalized,
        qber=total_qber,
        threshold=error_threshold,
        params=params,
        step=step,
        attack_intensity=adversary.intensity,
        filter_effectiveness=filter_reduction
    )
    
    # Enhanced termination condition
    done = evaluate_enhanced_termination(total_qber, error_threshold, step, key_rate_normalized)
    
    return observations, reward, done

def compute_advanced_reward(key_rate: float, qber: float, threshold: float, 
                          params: Dict[str, float], step: int, attack_intensity: float,
                          filter_effectiveness: float) -> float:
    """
    Advanced reward function based on recent QRL research
    Incorporates multiple objectives with careful balancing
    """
    reward = 0.0
    
    # Primary objective: Secure key generation (exponential scaling for better gradients)
    if key_rate > 0:
        # Base key rate reward with exponential bonus for high performance
        reward += 2000 * key_rate
        
        # Performance milestones with increasing rewards
        milestones = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for i, milestone in enumerate(milestones):
            if key_rate >= milestone:
                reward += (500 * (i + 1))  # Increasing rewards
    
    # Security objective: QBER management with smooth transitions
    qber_ratio = qber / max(threshold, 1e-6)
    if qber <= threshold:
        # Reward for staying below threshold, higher reward for lower QBER
        security_bonus = 800 * (1 - qber_ratio)**2  # Quadratic bonus
        reward += security_bonus
        
        # Extra bonus for maintaining very low QBER
        if qber < threshold * 0.5:
            reward += 600
        if qber < threshold * 0.3:
            reward += 800
    else:
        # Smooth penalty to avoid cliff effects
        excess_penalty = 1200 * (qber_ratio - 1)**2
        reward -= excess_penalty
    
    # Adaptive defense reward: Reward effective filtering
    if attack_intensity > 0.05:
        defense_bonus = 400 * filter_effectiveness * attack_intensity
        reward += defense_bonus
    
    # Operational efficiency rewards
    # Basis selection efficiency (peaked at 0.5)
    basis_efficiency_score = 200 * (1 - 4 * (params["basis_bias"] - 0.5)**2)
    
    # Privacy amplification efficiency (moderate values preferred)
    privacy_efficiency_score = 150 * np.exp(-5 * (params["privacy_strength"] - 0.25)**2)
    
    # Filter efficiency (no unnecessary filtering when no attack)
    if attack_intensity < 0.02:
        filter_penalty = -100 * params["filter_aggressiveness"]
    else:
        filter_penalty = 0
    
    reward += basis_efficiency_score + privacy_efficiency_score + filter_penalty
    
    # Stability reward: Consistent performance over time
    if step > 1000 and qber < threshold * 0.8:
        stability_bonus = 300
        reward += stability_bonus
    
    # Learning progress reward (early training encouragement)
    if step < 50000:
        progress_factor = min(1.0, step / 10000)
        reward += 200 * progress_factor
    
    # Base operational reward
    reward += 150
    
    # Catastrophic failure penalty
    if qber > 0.45 or key_rate == 0.0:
        reward -= 2000
    
    # Ensure reward is finite and reasonable
    reward = np.clip(reward, -8000, 15000)
    
    return reward

def evaluate_enhanced_termination(qber: float, threshold: float, step: int, key_rate: float) -> bool:
    """Enhanced episode termination with research considerations"""
    if step < 2000:  # Extended grace period for learning
        return False
    
    # Terminate on sustained catastrophic failure
    if qber > min(0.4, threshold * 3.0) and key_rate == 0:
        return True
    
    # Terminate on complete communication breakdown
    if qber > 0.48 and step > 5000:  # Near theoretical limit
        return True
    
    # Early termination for very long episodes with poor performance
    if step > 8000 and key_rate < 0.01 and qber > threshold * 1.5:
        return True
        
    return False

# Reset function for environment
def reset_simulation():
    """Reset all simulation state"""
    global adversary, hardware
    adversary = AdversarialState()
    hardware.reset()