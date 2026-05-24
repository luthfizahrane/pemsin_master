"""
DQN Reinforcement Learning - Part 1: Setup & Code Explanation
Tutorial interaktif untuk memahami Deep Q-Learning pada CartPole-v1

Author: Comprehensive DQN Tutorial
Date: 2026
"""

print("="*80)
print("DEEP Q-LEARNING (DQN) - PART 1: SETUP & PENJELASAN KODE")
print("="*80)
print()

# ============================================================================
# SECTION 1: INSTALL DEPENDENCIES
# ============================================================================
print("[1/5] Installing dependencies...")

import subprocess
import sys

packages = ['gymnasium[classic_control]']
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✓ Dependencies installed")
print()

# ============================================================================
# SECTION 2: IMPORT LIBRARIES
# ============================================================================
print("[2/5] Importing libraries...")

import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, deque
from itertools import count
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime

# Setup
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    try:
        from IPython import display
    except:
        pass

plt.ion()

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"✓ All libraries imported")
print(f"✓ Device: {device}")
print()

# ============================================================================
# SECTION 3: ENVIRONMENT SETUP
# ============================================================================
print("[3/5] Setting up environment...")

env = gym.make("CartPole-v1")
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

print(f"✓ Environment: CartPole-v1")
print(f"✓ State size: {n_observations} dimensions")
print(f"✓ Action space: {n_actions} actions (left/right)")
print()

# ============================================================================
# SECTION 4: TABEL INTERPRETASI HYPERPARAMETER
# ============================================================================
print("[4/5] Hyperparameter Interpretation Table")
print()

hyperparams_info = {
    'BATCH_SIZE': {
        'default': 128,
        'range': '32-256',
        'function': 'Jumlah sampel transitions per update',
        'effect': '↑ Stable tapi lambat; ↓ Cepat tapi noisy'
    },
    'GAMMA (γ)': {
        'default': 0.99,
        'range': '0.90-0.99',
        'function': 'Discount factor untuk future rewards',
        'effect': '↑ Fokus long-term; ↓ Fokus short-term'
    },
    'EPS_START': {
        'default': 0.9,
        'range': '0.5-1.0',
        'function': 'Probabilitas awal exploration',
        'effect': '↑ Explore banyak; ↓ Exploit early'
    },
    'EPS_END': {
        'default': 0.01,
        'range': '0.01-0.1',
        'function': 'Probabilitas akhir exploration',
        'effect': '↑ Tetap explore; ↓ Pure exploitation'
    },
    'EPS_DECAY': {
        'default': 2500,
        'range': '1000-10000',
        'function': 'Kecepatan decay epsilon',
        'effect': '↑ Decay lambat; ↓ Decay cepat'
    },
    'TAU': {
        'default': 0.005,
        'range': '0.001-0.01',
        'function': 'Update rate target network',
        'effect': '↑ Lebih stabil; ↓ Lebih responsif'
    },
    'LR (Learning Rate)': {
        'default': 3e-4,
        'range': '1e-4-1e-3',
        'function': 'Learning rate optimizer',
        'effect': '↑ Cepat convergence; ↓ Stabil tapi slow'
    }
}

df_hyperparams = pd.DataFrame([
    {
        'Parameter': k,
        'Default': v['default'],
        'Range': v['range'],
        'Function': v['function']
    }
    for k, v in hyperparams_info.items()
])

print(df_hyperparams.to_string(index=False))
print()
print()

# ============================================================================
# SECTION 5: PENJELASAN KODE PENTING
# ============================================================================
print("[5/5] Penjelasan Kode Penting")
print()

explanations = """
═══════════════════════════════════════════════════════════════════════════════
3.1 STATE REPRESENTATION
═══════════════════════════════════════════════════════════════════════════════
CartPole-v1 memberikan state sebagai VEKTOR 4 DIMENSI:

  state = [position, velocity, angle, angular_velocity]
  
  • position: posisi cart (-2.4 to 2.4)
  • velocity: kecepatan cart
  • angle: sudut pole dari vertikal (-0.42 to 0.42 radian)
  • angular_velocity: kecepatan sudut pole

Contoh state real:
"""
print(explanations)
print(f"  {state}")
print()

explanations2 = """
═══════════════════════════════════════════════════════════════════════════════
3.2 ACTION SELECTION (EPSILON-GREEDY POLICY)
═══════════════════════════════════════════════════════════════════════════════
Agent memilih action berdasarkan exploration-exploitation tradeoff:

  ε(t) = ε_end + (ε_start - ε_end) × e^(-t / ε_decay)
  
  if random() > ε:
      action = argmax(Q(state, a))  ← Exploitation: pilih best action
  else:
      action = random()              ← Exploration: random action

Visualisasi:
  • Awal training: ε tinggi → lebih banyak random action
  • Akhir training: ε rendah → lebih banyak learned policy
  
═══════════════════════════════════════════════════════════════════════════════
3.3 REWARD STRUCTURE
═══════════════════════════════════════════════════════════════════════════════
  • +1 point: untuk setiap timestep pole tetap upright
  • Episode ends: ketika angle > 0.21 radian ATAU cart > 2.4 units
  • Goal: Maximize total reward dalam episode (max 500)

═══════════════════════════════════════════════════════════════════════════════
3.4 Q-VALUE DAN BELLMAN EQUATION
═══════════════════════════════════════════════════════════════════════════════
Fungsi Q(s,a) = expected return jika ambil action a di state s

Bellman Equation:
  Q(s,a) ≈ r + γ × max_a' Q(s', a')
  
  Dimana:
  • r = immediate reward
  • γ = discount factor (0.99)
  • s' = next state
  • max_a' = best next action

Training update:
  Loss = (Q_predicted - Q_target)²
       = (Q(s,a) - [r + γ × max_a' Q_target(s', a')])²

═══════════════════════════════════════════════════════════════════════════════
3.5 NEURAL NETWORK ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════
Simple fully-connected network:

  Input(4) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(2) → Output
   [state]                                                      [Q-values]

  • Input: 4-dimensional state
  • Hidden layers: 128 neurons each with ReLU activation
  • Output: Q-value untuk 2 actions (left/right)

═══════════════════════════════════════════════════════════════════════════════
3.6 TRAINING LOOP
═══════════════════════════════════════════════════════════════════════════════
Pseudocode:

  for each episode:
      reset environment
      for each timestep:
          select_action(state) using ε-greedy
          execute action → get (next_state, reward)
          store transition (s, a, s', r) in replay memory
          
          sample batch dari memory
          compute Q_target = r + γ × max_a' Q_target(s')
          compute loss = MSE(Q_predicted, Q_target)
          
          backprop & update policy_net weights
          soft_update target_net: θ' ← τθ + (1-τ)θ'

Key insight: Experience replay decorrelates transitions → stable learning

═══════════════════════════════════════════════════════════════════════════════
"""
print(explanations2)

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("✓ PART 1 SELESAI")
print("="*80)
print()
print("Untuk melanjutkan:")
print("  1. Jalankan PART 2: Baseline Training")
print("  2. Jalankan PART 3: Modified Training & Analysis")
print()
print("="*80)
