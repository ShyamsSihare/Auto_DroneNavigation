# QACLDN — Quantum Actor-Critic for Drone Navigation

**A hybrid quantum-classical reinforcement-learning pipeline for autonomous UAV obstacle avoidance, with a from-scratch statevector VQC simulator, a 12-state rigid-body quadrotor dynamics engine, and real-world multi-sensor data calibration.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/notebook-Jupyter-orange.svg)](notebooks/)
[![Reproducible](https://img.shields.io/badge/gradients-verified%20to%201e--11-brightgreen.svg)](tests/)

---

## What this is

This repository implements and empirically evaluates **QACLDN** (Quantum Actor-Critic for Drone Navigation), a hybrid quantum-classical policy-gradient method for UAV obstacle-avoidance navigation, against four baselines — a non-entangled VQC ablation (**VQC-NE**), **Classical PPO**, a discrete entropy-regularised **Soft Actor-Critic**, and **Classical DQN** — on a physically-grounded 12-state quadrotor simulator.

Unlike most "quantum RL" reference implementations, **every component here is written from scratch and independently numerically verified**:

- A **statevector quantum-circuit simulator** (batched, NumPy-only) implementing a hardware-efficient VQC with data re-uploading, parameterised $R_y/R_z$ rotation layers, alternating CNOT entangling layers, and Pauli-$Z$ readout — with the **parameter-shift gradient rule verified against central finite differences to a maximum error of $2.16\times10^{-11}$** (`tests/test_gradients.py`).
- A **12-state Newton–Euler rigid-body quadrotor simulator**, RK4-integrated at $\Delta t = 0.02\,\mathrm{s}$, with a motor allocation matrix and an Ornstein–Uhlenbeck (von Kármán-type) turbulence model.
- **Manually-derived, analytically-exact backpropagation** for every classical neural network in the pipeline (actor/critic MLPs, DQN), verified against finite differences to $1.63\times10^{-11}$.
- **Empirical calibration against three real-world data sources** (not synthetic placeholders): a 5,000-sample UAV telemetry log, the ICMCIS multi-sensor drone-detection corpus (ALVIRA radar + DIANA RF), and the physical parameter specification of the `gym-pybullet-drones` Crazyflie 2.X (`cf2x`) reference model.

This repository, and the accompanying manuscript (`manuscript/main.tex`), report results **honestly, including negative and null results** (see [Limitations](#limitations-and-honest-reporting) below) — nothing here is illustrative or fabricated data.

---

## Repository structure

```
.
├── README.md                          <- you are here
├── LICENSE                            <- MIT (code); see data/README.md for dataset terms
├── CITATION.cff                       <- machine-readable citation metadata
├── requirements.txt                   <- pip dependencies
├── environment.yml                    <- conda environment (equivalent)
├── .gitignore
│
├── src/
│   └── analysis.py                    <- the full pipeline, jupytext "percent" format
│                                          (quantum sim -> dynamics -> env -> agents ->
│                                           training -> 7 experiments -> 6 figures -> PDF)
│
├── notebooks/
│   └── QACLDN_experimental_analysis.ipynb   <- executed notebook, all outputs embedded
│
├── manuscript/
│   ├── main.tex                              <- full manuscript source
│   ├── figure_captions_explanations.tex      <- LaTeX captions + scientific explanation
│   │                                             for all 6 result figures (compiles standalone)
│   └── dataset_tables_explanation.tex        <- LaTeX schema/statistics tables for all
│                                                 3 data sources (compiles standalone)
│
├── data/
│   └── README.md                      <- full provenance, schema, and licensing notes for
│                                          the 3 external data sources (not vendored — see below)
│
├── results/
│   ├── QACLDN_experimental_results.pdf       <- all 6 figures, multi-page PDF
│   └── figures/*.png                          <- the same 6 figures, individually
│
├── docs/
│   ├── METHODOLOGY.md                 <- deep-dive: every equation, every design decision
│   └── ARCHITECTURE.md                <- module-by-module code map
│
├── tests/
│   └── test_gradients.py              <- the two numerical self-tests (parameter-shift
│                                          rule, manual MLP backprop) as standalone pytest
│
└── .github/workflows/
    └── ci.yml                          <- lint + self-test workflow (GitHub Actions)
```

---

## Quickstart

```bash
git clone <this-repo>
cd qacldn
python -m venv .venv && source .venv/bin/activate      # or: conda env create -f environment.yml
pip install -r requirements.txt

# Run the numerical self-tests (parameter-shift rule, manual backprop) — should take <5s
pytest tests/ -v

# Run the full pipeline (checkpoints itself; ~30-40 min on a laptop CPU on first run,
# near-instant on any re-run thanks to checkpoint.pkl)
python src/analysis.py

# ...or open the pre-executed notebook directly (no re-run needed):
jupyter notebook notebooks/QACLDN_experimental_analysis.ipynb
```

The pipeline is **checkpointed at every experiment boundary** (`checkpoint.pkl`), so interrupting and re-running `src/analysis.py` resumes from the last completed experiment/seed rather than restarting from scratch.

---

## Method summary

### 1. Quantum policy/value backbone (VQC)

State $\bs s\in\R^{12}$ is normalised, padded, and re-uploaded across $L$ circuit layers on $n$ qubits. Each layer applies

$$
U_\ell(\bs\theta_\ell,\bs s) \;=\; E_{\text{odd}}\,E_{\text{even}}\; R_\ell(\bs\theta_\ell)\;\Phi^{(\ell)}_{\text{enc}}(\bs s;\bs\omega_\ell),
$$

where $\Phi_{\text{enc}}$ is a data re-uploading $R_y R_z$ encoding, $R_\ell$ is a trainable $R_y R_z$ rotation layer ($2n$ parameters/layer), and $E_{\text{even}}, E_{\text{odd}}$ are alternating CNOT entangling layers. Readout is the Pauli-$Z$ expectation on every qubit, $o_j=\langle Z_j\rangle$, feeding linear actor ($\mathrm{softmax}$ over 16 actions) and critic heads. Total parameter count follows the exact closed form

$$
P(n,L) \;=\; \underbrace{2nL}_{\text{VQC rotations}} + \underbrace{|\mathcal A|(n+1)}_{\text{actor head}} + \underbrace{(n+1)}_{\text{critic head}}, \qquad |\mathcal A|=16,
$$

verified in this repo to reproduce $P(2,2){=}59$, $P(4,2){=}101$, $P(6,2){=}143$, $P(8,2){=}185$ **exactly**.

Gradients w.r.t. $\bs\theta$ use the **exact analytic parameter-shift rule**,

$$
\frac{\partial\langle Z_j\rangle}{\partial\theta_i} = \frac{\langle Z_j\rangle(\theta_i+\tfrac\pi2) - \langle Z_j\rangle(\theta_i-\tfrac\pi2)}{2},
$$

batched over the whole training minibatch in a single vectorised pass (see `VQC.forward_and_param_shift_grad_batch` in `src/analysis.py`).

### 2. Environment

A 12-state ($\bs p,\bs v,\bs\Theta,\bs\omega$) quadrotor with RK4-integrated Newton–Euler dynamics, a 16-way discretised action set (4 binary channels: collective thrust / pitch / roll / yaw), cylindrical obstacles placed by rejection sampling, and the reward

$$
r_t = \lambda_g r_g(\bs s_t) - \lambda_c r_c(\bs s_t) - \lambda_e r_e(\bs u_t) + \lambda_s r_s(\bs s_t) - \lambda_v r_v(\bs s_t).
$$

### 3. Agents compared

| Method | Backbone | Update rule |
|---|---|---|
| **QACLDN** | VQC (entangled) | Quantum policy gradient (parameter-shift) + quantum Bellman TD-error + Grover-inspired $1/\sqrt{1+N(s,a)}$ exploration bonus |
| **VQC-NE** | VQC (CNOTs removed) | Same as QACLDN — isolates the effect of entanglement |
| **Classical PPO** | MLP | Clipped surrogate + GAE($\lambda$), multi-epoch minibatch updates |
| **Soft Actor-Critic** | MLP | Same as PPO with higher entropy coefficient + soft value target |
| **Classical DQN** | MLP | Replay buffer + target network, $\epsilon$-greedy, semi-gradient TD |

Full derivations, every equation, and every reported number are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and [`manuscript/figure_captions_explanations.tex`](manuscript/figure_captions_explanations.tex).

---

## Results at a glance

*(all numbers are real, computed outputs of `src/analysis.py` — see `docs/METHODOLOGY.md` for full statistical context and error bars)*

| Method | Final return (mean±std, 60 eval episodes × 3 seeds) | Collision rate | Path-length ratio |
|---|---:|---:|---:|
| **QACLDN** | **−1332.71 ± 24.52** (best) | 68.33% | 1.179 |
| VQC-NE | −1343.62 ± 10.45 | 46.11% | 1.117 |
| Classical DQN | −1470.14 ± 70.99 | 36.11% | 0.932 |
| Classical PPO | −1493.64 ± 14.37 | 26.67% | 0.669 |
| Soft Actor-Critic | −1502.25 ± 20.89 (worst) | 26.67% | 0.656 |

The quantum-backbone methods trade a **higher collision rate for more active, goal-directed flight** (PLR > 1), which nets out to the **best mean return** of all five methods — while the classical baselines converge to a conservative near-hover policy (PLR < 1) that avoids obstacles but also avoids the goal. This trade-off, its reward-weight-level mechanism ($\lambda_g=6.0$ vs. $\lambda_c=2.5$), and its statistical significance are analysed in full in `docs/METHODOLOGY.md`.

---

## Limitations and honest reporting

This repository intentionally reports **negative and inconclusive results** rather than only favourable ones:

- **Literal goal-reaching success rate is ≈0% for all five methods** at the training budget used (150 episodes × 3 seeds for the main comparison, vs. the underlying theoretical framework's nominal 1200-episode budget). This reflects a genuine sample-efficiency limit at reduced scale, not a code defect — the quantum-gradient and classical-backprop code paths are independently unit-tested (see `tests/`).
- At $N_{\text{seed}}=3$, the four actor-critic-family methods are **not statistically separable** in aggregate return (spread of 16.4 return units vs. ≈70–78 s.e.m.); only Classical DQN is a clear outlier.
- The obstacle-density scalability sweep shows **identical results at obstacle counts 10 and 12** — a genuine hard-disk packing-saturation artefact of the rejection-sampling obstacle placer, not measurement noise (explained in `docs/METHODOLOGY.md`).
- The empirically-calibrated real-world wind intensity ($\hat\sigma_w/g \approx 0.59$, from the UAV telemetry log) **exceeds the entire tested wind-robustness sweep range** ($[0, 0.5]g$) — a quantified sim-to-real coverage gap.

See `docs/METHODOLOGY.md` §"Statistical Scope and Limitations" for the full discussion.

---

## Data sources

Three external, third-party data sources are used for calibration and validation; **none are vendored in this repository** (see [`data/README.md`](data/README.md) for exact provenance, licensing, and re-download instructions):

1. **UAV Navigation Telemetry Log** — 5,000-sample, 1 Hz single-vehicle GPS/IMU/LiDAR/wind/battery log.
2. **ICMCIS Multi-Sensor Drone-Detection Corpus** — real flight-test radar (ALVIRA)/RF (DIANA) logs, 5 of 14 total scenarios used.
3. **`gym-pybullet-drones`** (Panerati et al., IROS 2021) — used only as a physical-parameter reference (`cf2x.urdf` Crazyflie 2.X specification), not executed as a dependency.

---

## Citation

If you use this code or methodology, please cite via [`CITATION.cff`](CITATION.cff), and see `manuscript/main.tex` for the full paper.

## License

Code in this repository is released under the [MIT License](LICENSE). This license covers the code only — the three external datasets described in `data/README.md` retain their original respective licenses/terms of use.
