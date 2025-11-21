# Task-Specific Sharpness-Aware O-RAN Resource Management using Multi-Agent RL

Official PyTorch implementation of the paper:

**“Task-Specific Sharpness-Aware O-RAN Resource Management using Multi-Agent Reinforcement Learning”**  
by *Fatemeh Lotfi, Hossein Rajoli, and Fatemeh Afghah* (IEEE Transactions on Machine Learning in Communications and Networking, 2025).

---

## 🚀 TL;DR

- We study **joint network slicing and resource block (RB) allocation** in an **O-RAN** architecture with multiple Distributed Units (DUs) modeled as **multi-agent RL (MARL) actors** and a global critic at the near-RT RIC. :contentReference[oaicite:0]{index=0}  
- We enhance **Soft Actor-Critic (SAC)** with **Sharpness-Aware Minimization (SAM)** and propose **Task-Aware SAM (TA-SAM)**:
  - SAM is applied **selectively** to actor networks based on **TD-error variance** (only “hard”/unstable agents get regularized). :contentReference[oaicite:1]{index=1}  
  - A **dynamic ρ scheduling** controls the perturbation radius over training, improving the exploration–exploitation trade-off.
- We show up to **≈22% improvement in resource allocation efficiency and QoS satisfaction** over conventional DRL baselines across diverse traffic and slice profiles. :contentReference[oaicite:2]{index=2}  

If you care about **robust, generalizable DRL for O-RAN resource management**, this repo is for you.

---

## 📄 Paper

> **Task-Specific Sharpness-Aware O-RAN Resource Management using Multi-Agent Reinforcement Learning**  
> IEEE Transactions on Machine Learning in Communications and Networking (TMLCN), 2025.

- PDF (preprint): **[https://arxiv.org/abs/2511.15002]**
- IEEE Xplore: **[https://ieeexplore.ieee.org/document/11260483]**

---

## 🔍 Method Overview

We consider an O-RAN system with:

- **Three slices**: eMBB, mMTC, URLLC, each with its own QoS metrics and priorities. :contentReference[oaicite:3]{index=3}  
- **Multiple DUs**, each acting as a **local MARL agent (actor)** controlling RB allocation for its users.
- A **global critic (xApp)** in the near-RT RIC that aggregates experience and updates all actors. :contentReference[oaicite:4]{index=4}  

Key ideas:

1. **MARL + SAC architecture**
   - Each DU = one actor.
   - A centralized global critic runs at the near-RT RIC.
   - SAC is used for continuous action learning; actions are later **thresholded** to binary RB allocation decisions for slicing + scheduling. :contentReference[oaicite:5]{index=5}  

2. **Task-Aware SAM (TA-SAM)**
   - **SAM in the critic**: smooths the loss landscape and stabilizes value estimation across heterogeneous traffic and slice conditions. :contentReference[oaicite:6]{index=6}  
   - **Selective SAM in the actors**:
     - We compute **TD-error variance** per actor as a proxy for task/environment complexity.
     - SAM is applied **only if** the TD-error variance exceeds a threshold.
   - This avoids wasting compute on easy actors and focuses regularization on unstable regions.

3. **Dynamic ρ scheduling**
   - ρ starts **large** → encourages exploration in sharp/unstable areas.
   - ρ is **decayed over time** → shifts toward exploitation and fine-tuning near flatter minima. :contentReference[oaicite:7]{index=7}  

---

## 🧱 Repository Structure

```text
TA-SAM-MARL-ORAN/
├── configs/
│   ├── tasam_dynamic_rho.yaml        # Main config for TA-SAM (dynamic ρ)
│   ├── sac_baseline.yaml             # Plain SAC baseline
│   ├── l2_baseline.yaml              # SAC + L2 regularization
│   └── ablation_actor_critic_sam.yaml
├── envs/
│   ├── oran_env.py                   # O-RAN simulator (slices, QoS, RBs, traffic)
│   └── traffic_models.py             # Traffic + mobility generation
├── marl/
│   ├── agents/
│   │   ├── sac_actor.py
│   │   ├── sac_critic.py
│   │   └── tasam_actor_wrapper.py    # Selective SAM logic + TD-error variance
│   ├── sam/
│   │   ├── sam_optimizer.py          # Generic SAM implementation
│   │   └── tasam_scheduler.py        # Dynamic ρ scheduling
│   ├── memory/
│   │   └── replay_buffer.py
│   ├── trainer.py                    # Main MARL training loop
│   └── utils.py
├── scripts/
│   ├── train_tasam.sh
│   ├── train_baselines.sh
│   └── eval_qos.sh
├── experiments/
│   ├── logs/
│   └── results/
├── figures/                          # Plots reproducing main paper figures
├── requirements.txt
├── README.md
└── LICENSE

---

## ⚙️ Installation
git clone https://github.com/FLotfiGit/TA-SAM-MARL-ORAN.git
cd TA-SAM-MARL-ORAN

# (recommended) create a virtual env or conda env
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt


## Minimum stack:

Python ≥ 3.9

PyTorch ≥ 2.0

NumPy, SciPy

Matplotlib / Seaborn (for plotting)

YAML / OmegaConf / hydra-core (if you use configs)

## 🏃 Quick Start: Training TA-SAM MARL

## Basic training command:

python -m marl.trainer \
  --config configs/tasam_dynamic_rho.yaml \
  --exp_name tasam_dynamic_rho_run1


## Key config fields:

num_dus: number of DUs / agents (default: 6)

num_slices: number of slices (default: 3 – eMBB, mMTC, URLLC)

use_sam_critic: true / false

use_sam_actor: true / false

td_var_threshold: TD-error variance threshold for enabling SAM on an actor

rho_init, rho_final, rho_decay_steps: schedule for ρ

train_steps, eval_interval, batch_size, buffer_size

---

## 📊 Reproducing Main Results

We evaluate the following variants (as in the paper):

** No-SAM: vanilla SAC MARL.**

** L2-reg: SAC + L2 weight regularization.**

** Actor-SAM: SAM only on actor networks. **

** Critic-SAM: SAM only on the global critic.**

** Both-SAM (TA-SAM): SAM on critic + selective SAM on actors (our full method). **

---

## Example runs:

# Vanilla SAC
python -m marl.trainer --config configs/sac_baseline.yaml --exp_name sac_baseline

# SAC + L2
python -m marl.trainer --config configs/l2_baseline.yaml --exp_name sac_l2

# Critic-only SAM
python -m marl.trainer --config configs/critic_sam.yaml --exp_name critic_sam

# Actor-only SAM
python -m marl.trainer --config configs/actor_sam.yaml --exp_name actor_sam

# Full TA-SAM (both + selective actor SAM + dynamic ρ)
python -m marl.trainer --config configs/tasam_dynamic_rho.yaml --exp_name tasam_full


## We log (per episode / training window):

Average cumulative reward

## Slice-level QoS metrics:

Latency, throughput, service availability, user density satisfaction 

RB utilization

TD-error variance per actor (to visualize where SAM is triggered)

Loss curvature proxies (Hessian / eigenvalue logging)

## plots:

Cumulative reward vs. training iterations for different ρ settings.

Boxplots of reward distributions across methods.

QoS satisfaction per slice (eMBB / URLLC / mMTC).

---

## 🧪 O-RAN Environment

The environment follows the system model in the paper: 

OFDM-based downlink, 20 MHz total bandwidth, RB bandwidth 200 kHz.

200 UEs, distributed across slices and DUs.

Rayleigh fading + inter-cell interference.

**State includes:**

Slice-level QoS metrics Ql

Number of UEs per slice N_u^l

Previous action a_{t-1}

**Actions:**

Joint RB-to-slice and RB-to-UE allocation using a high-dimensional binary vector (real outputs from SAC + thresholding).

**Reward:**

Nonlinear combination of QoS metrics via sigmoids + penalties for:

Exceeding RB capacity

Violating minimum QoS thresholds 

---

## 🧩 Coming Soon

I’m still cleaning up and refactoring the code for public release. Planned additions:

✅ Example configs for all baselines (SAC, L2, Actor-SAM, Critic-SAM, Both-SAM)

✅ Plotting scripts to recreate the main figures

🔜 Dockerfile / environment YAML for exact reproducibility

🔜 Integration notes for real O-RAN testbeds / RIC simulators

Stay tuned and ⭐ the repo if you want to track updates.

---

## 📚 Citation

If you use this code or ideas from the paper, please cite:

@ARTICLE{lotfi2025tasamoran,
  author={Lotfi, Fatemeh and Rajoli, Hossein and Afghah, Fatemeh},
  journal={IEEE Transactions on Machine Learning in Communications and Networking}, 
  title={Task-Specific Sharpness-Aware O-RAN Resource Management using Multi-Agent Reinforcement Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Resource management;Open RAN;Network slicing;Dynamic scheduling;Wireless networks;Adaptation models;Complexity theory;Real-time systems;Optimization;Deep reinforcement learning;Open RAN;Deep reinforcement learning;Network slicing and scheduling;Sharpness aware minimization;Multi agent reinforcement learning},
  doi={10.1109/TMLCN.2025.3634994}}

---

## 📬 Contact

For questions, feel free to reach out:

Fatemeh Lotfi – flotfi@clemson.edu

Or open an issue / discussion in this repo.
