# Harpy Walker RL: Stable Biped Locomotion using Reinforcement Learning

This project explores the feasibility of training a custom bipedal robot — Harpy — to walk stably using reinforcement learning (RL). Using both Proximal Policy Optimization (PPO) and Twin Delayed Deep Deterministic Policy Gradient (TD3), we investigate how morphology, torque constraints, and reward structure influence gait quality in MuJoCo simulations.

## Project Overview

- **Course:** CS 5180 - Reinforcement Learning  
- **Institution:** Northeastern University  
- **Authors:** Jorge Ortega, Patricia Meza  
- **Date:** April 2025  

The Harpy robot has an unusual structure: a heavy torso and massless legs, making it hard to control using traditional methods. Our goal was to test if RL algorithms could discover compensatory walking strategies in this challenging setup.

## Algorithms Used

- **Proximal Policy Optimization (PPO)**  
  Trained using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), with parallel environments and TensorBoard logging.

- **Twin Delayed Deep Deterministic Policy Gradient (TD3)**  
  Implemented from scratch using PyTorch. Custom actor-critic architectures with double Q-learning and target policy smoothing.

## Environments and Tools

- **Simulator:** [MuJoCo](https://mujoco.org/)
- **Environment:** [`Walker2d-v5`](https://gymnasium.farama.org/environments/mujoco/walker2d/)
- **Frameworks:** Gymnasium, PyTorch, Stable-Baselines3, TensorBoard
- **Hardware:** CPU for PPO / GPU (RTX 3060) for TD3

## Key Modifications

### Harpy-Inspired Morphology:
- Mass reduced to ~6 kg (from 24 kg)
- Torso mass = 4 kg; feet/thighs scaled down
- Torque limits: ±20 Nm (hips/knees), ±10 Nm (ankles)

### Reward Function:
\[
\text{Reward} = \text{healthy reward} + \text{forward velocity} - \text{control cost}
\]

Modified versions penalize excessive energy use and favor stable/slow gaits.

 **Video Demos:**
- [PPO forward gait (49s)](https://youtube.com/shorts/89PpLuFsB4k)  
- [TD3 forward gait (1 min)](https://youtu.be/DlBghiJG6jE)  
- [TD3 backward gait (20s)](https://youtu.be/yGeIv_q2mSc)

## 📈 Training Insights

- **PPO**: Good stability but suffered from noisy value loss and entropy collapse in some configurations. Could not modify MuJoCo model during parallel training.
- **TD3**: Achieved best reward and natural motion. Sensitive to torque limits and exploration noise but performed better with realistic physics.
