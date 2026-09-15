# Methodology — Full Scientific Reference

This document is the deep-dive companion to the top-level `README.md`. It gives every equation, every design decision, and every real computed number produced by `src/analysis.py`, organised the way a methods section of a paper would be. It intentionally repeats no fabricated or illustrative numbers — every value below was read directly from an executed run of the pipeline (`checkpoint.pkl` / notebook outputs).

## Table of contents

1. [Real-world data calibration](#1-real-world-data-calibration)
2. [Quantum circuit simulator](#2-quantum-circuit-simulator)
3. [Quadrotor rigid-body dynamics](#3-quadrotor-rigid-body-dynamics)
4. [Environment and reward function](#4-environment-and-reward-function)
5. [Agents](#5-agents)
6. [Experiments and results](#6-experiments-and-results)
7. [Statistical scope and limitations](#7-statistical-scope-and-limitations)

---

## 1. Real-world data calibration

Three real data sources ground the otherwise-synthetic simulation (full schema and provenance in `data/README.md`):

### 1.1 Wind turbulence model

Atmospheric disturbance is modelled as a stationary Ornstein–Uhlenbeck (OU) process,

$$
d\xi(t) = -\frac{1}{\tau_w}\xi(t)\,dt + \sigma_w\sqrt{\frac{2}{\tau_w}}\,dW(t), \qquad
\mathrm{Cov}[\xi(t)\xi(t')] = \sigma_w^2 e^{-|t-t'|/\tau_w},
$$

discretised exactly (not approximated) as

$$
\xi_{k+1} = a\,\xi_k + \sigma_w\sqrt{1-a^2}\,\mathcal N(0,1), \qquad a \equiv e^{-\Delta t/\tau_w}.
$$

The intensity $\sigma_w$ is read directly from the sample standard deviation of the logged `wind_speed` channel of the UAV telemetry dataset: $\hat\sigma_w = 5.768743\ \mathrm{m/s}$. The correlation time is recovered from the empirical autocorrelation function via

$$
\hat\tau_w = \left(-\frac{1}{\sum_{k=1}^{8}(k\Delta t)^2}\sum_{k=1}^{8}(k\Delta t)\ln\hat\rho_k\right)^{-1} = 0.63\ \mathrm{s},
$$

a factor of $3.17\times$ **shorter** than a commonly assumed textbook constant of $\tau_w = 2.0\ \mathrm{s}$ — i.e. real near-surface gust turbulence decorrelates roughly three times faster than the idealised assumption.

### 1.2 Radar path-loss calibration

The ICMCIS DIANA RF sensor's real detections ($N=2681$) are fit by ordinary least squares to the one-way path-loss model

$$
\mathrm{SNR}(R)\ [\mathrm{dB}] = A - 10\,n_{\mathrm{pl}}\log_{10}(R),
$$

giving $\hat n_{\mathrm{pl}} = 1.994$, $\hat A = 64.83\ \mathrm{dB}$. This sits within 0.3% of the theoretical one-way free-space exponent $n_{\mathrm{pl}}=2$ (as opposed to $n_{\mathrm{pl}}=4$ for a two-way monostatic radar return, $P_r \propto R^{-4}/(4\pi)^3$), which — derived from the data itself, not assumed — confirms DIANA is functioning as a passive RF interceptor rather than an active radar.

### 1.3 Sensor noise floors

- LiDAR range-increment noise: $\hat\sigma_{\mathrm{LiDAR}} = 28.80\ \mathrm{m}$ (first-difference std of the `lidar_distance` channel, corrected by $\sqrt2$).
- IMU acceleration-increment noise: $\hat\sigma_{\mathrm{IMU}} = 0.828\ \mathrm{m/s^2}$.

### 1.4 Physical parameter reference (`gym-pybullet-drones`)

The `cf2x.urdf` Crazyflie 2.X specification (mass $0.027\ \mathrm{kg}$, arm $0.0397\ \mathrm{m}$, thrust-to-weight $2.25$) is used — not executed, only its URDF parameters read — as an authoritative physical reference, uniformly rescaled ($\times33$ mass, $\times5.04$ arm length) to the larger vehicle class assumed by the simulator. The exact hover command follows in closed form from the thrust-to-weight ratio $\lambda_{T/W}=2.5$ actually used:

$$
u_{\mathrm{hover}} = \frac{1}{\sqrt{\lambda_{T/W}}} = \frac{1}{\sqrt{2.5}} \approx 0.632,
$$

which centres the 16-point discretised action set exactly on a true dynamic equilibrium.

---

## 2. Quantum circuit simulator

A from-scratch, batched, NumPy-only statevector simulator (`class VQC` in `src/analysis.py`). No Qiskit/PennyLane dependency — single-qubit gates are applied by reshaping the $2^n$-dimensional statevector into a `(batch, left, 2, right)` tensor and contracting the $2\times2$ gate on the middle axis, giving $O(B\cdot2^n)$ cost per gate rather than $O(4^n)$ for an explicit operator matrix.

**Circuit structure** per layer $\ell = 1,\dots,L$:

$$
U_\ell(\bs\theta_\ell,\bs s) = E_{\text{odd}}\,E_{\text{even}}\,R_\ell(\bs\theta_\ell)\,\Phi^{(\ell)}_{\text{enc}}(\bs s),
$$

- $\Phi^{(\ell)}_{\text{enc}}$: data re-uploading — one $R_y$, one $R_z$ per qubit, with rotation angle $\pi\cdot\tilde s_i$ where $\tilde s_i\in[0,1]$ is a (layer-shifted) component of the normalised, padded state vector. Because the encoding index cycles with the layer index, every layer sees a different view of the full state — a genuine re-uploading circuit, not a single fixed embedding.
- $R_\ell$: trainable $R_y R_z$ rotation, $2n$ free parameters per layer.
- $E_{\text{even}}, E_{\text{odd}}$: alternating CNOT entangling layers (even pairs, then odd pairs).
- Readout: Pauli-$Z$ expectation on every qubit, $o_j = \langle Z_j\rangle = P(0)-P(1) \in [-1,1]$.

**Parameter count** (exact closed form, verified against measured counts):

$$
P(n,L) = \underbrace{2nL}_{\text{VQC rotations}} + \underbrace{16(n+1)}_{\text{actor head}} + \underbrace{(n+1)}_{\text{critic head}}.
$$

| $n$ | $L$ | $P(n,L)$ formula | Measured |
|---|---|---|---|
| 2 | 2 | $8+48+3$ | **59** |
| 4 | 2 | $16+80+5$ | **101** |
| 6 | 2 | $24+112+7$ | **143** |
| 8 | 2 | $32+144+9$ | **185** |

**Gradient rule.** For any single-qubit-rotation-generated parameter, the parameter-shift rule gives the *exact* (not approximate) analytic gradient:

$$
\frac{\partial\langle Z_j\rangle}{\partial\theta_i} = \frac{\langle Z_j\rangle(\theta_i+\tfrac\pi2) - \langle Z_j\rangle(\theta_i-\tfrac\pi2)}{2}.
$$

This is computed **batched** across the whole training minibatch in `VQC.forward_and_param_shift_grad_batch` — $2P$ batched forward passes total per gradient step (two shifts per parameter), each processing the entire batch via vectorised tensor contractions, rather than $2PB$ individual circuit simulations. This single optimisation is what makes VQC-in-the-loop policy-gradient training tractable at all on a CPU.

**Verification.** Parameter-shift gradients are checked against central finite differences ($\epsilon=10^{-5}$) in `tests/test_gradients.py`: maximum observed error $2.16\times10^{-11}$ (well within double-precision floating-point tolerance).

**Depolarising noise model.** A Monte-Carlo unravelling of the single-qubit depolarising channel,

$$
\rho \mapsto (1-p)\rho + \frac{p}{3}(X\rho X+Y\rho Y+Z\rho Z),
$$

which contracts the Bloch vector uniformly, $\bs r \mapsto (1-\tfrac{4p}{3})\bs r$, is applied stochastically after every entangling layer for the NISQ noise-resilience experiment (§6.5).

---

## 3. Quadrotor rigid-body dynamics

Twelve-state vehicle $\bs s = [\bs p,\bs v,\bs\Theta,\bs\omega]\in\R^{12}$ (position, velocity, ZYX Euler angles, body angular rates). Newton–Euler equations:

$$
m\ddot{\bs p} = -mg\bs e_3 + R(\phi,\theta,\psi)\bs F_T + \bs F_w(t), \qquad
\bs J\dot{\bs\omega} = -\bs\omega\times\bs J\bs\omega + \bs\tau_T + \bs\tau_d(t),
$$

integrated with classical fourth-order Runge–Kutta at $\Delta t=0.02\ \mathrm{s}$. A motor allocation matrix $M$ maps four squared motor commands $\bs u^2\in[0,1]^4$ to $[T_{\text{total}},\tau_\phi,\tau_\theta,\tau_\psi]$; per-motor thrust coefficient $c_T=\lambda_{T/W}mg/4$ is derived from the target thrust-to-weight ratio (§1.4), and torque coefficient $c_Q=0.015\,c_T$ (typical propeller drag/thrust ratio). A small proportional-derivative inner-loop stabiliser ($K_{p,\text{att}}=0.55$, $K_{d,\text{att}}=0.12$) is included to prevent the coarse 16-action discretisation from producing an uncontrollable tumble, consistent with standard low-level attitude stabilisation on real flight controllers.

Numerical safety: every state update is passed through `np.nan_to_num` and velocity/angular-rate clipping, guarding against the rare divergent trajectory poisoning a training batch.

---

## 4. Environment and reward function

$$
r_t = \lambda_g\,r_g(\bs s_t) - \lambda_c\,r_c(\bs s_t) - \lambda_e\,r_e(\bs u_t) + \lambda_s\,r_s(\bs s_t) - \lambda_v\,r_v(\bs s_t),
$$

with

$$
r_g = \exp\!\left(-\frac{\lVert\bs p-\bs p_g\rVert^2}{\sigma_g^2}\right) - 1,\quad
r_c = \mathbf 1[\lVert\bs p-\bs p_{\text{obs}}\rVert\le d_{\text{safe}}],\quad
r_e = \frac{\lVert\bs u\rVert^2}{4},\quad
r_s = -\lVert\bs\omega\rVert^2,\quad
r_v = \max(0,\lVert\bs v\rVert-v_{\max})^2.
$$

**Recalibrated weights** (see §7 for why, relative to an initial configuration): $\lambda_g=6.0$, $\lambda_c=2.5$, $\lambda_e=0.35$, $\lambda_s=0.06$, $\lambda_v=0.20$, $\sigma_g=2.2\ \mathrm{m}$, $d_{\text{safe}}=0.55\ \mathrm{m}$. Obstacles are cylindrical, placed by rejection sampling (up to 500 attempts, minimum pairwise separation $1.3\ \mathrm{m}$) inside a $2.6\ \mathrm{m}$-half-extent arena. The 16-action set is built from 4 binary control channels (collective thrust, pitch, roll, yaw), each a $\pm\delta$ perturbation ($\delta=0.22$) around the closed-form hover command $u_{\text{hover}}\approx0.632$ (§1.4).

---

## 5. Agents

| Agent | Backbone | Policy update |
|---|---|---|
| **QACLDN** | VQC, entangled | Quantum parameter-shift policy gradient + quantum Bellman TD-error $\delta_t = \tilde r_t+\gamma V(s_{t+1})-V(s_t)$ (where $\tilde r_t=r_t+\eta\cdot\text{bonus}$) + Grover-inspired exploration bonus $1/\sqrt{1+N(s,a)}$ |
| **VQC-NE** | VQC, CNOTs removed | Identical update rule — isolates the causal effect of entanglement |
| **Classical PPO** | MLP (tanh hidden, linear heads) | GAE($\lambda=0.92$) advantages, clipped surrogate objective, 3 epochs/minibatch update |
| **Soft Actor-Critic** | MLP | Same as PPO, $3\times$ entropy coefficient, soft value target |
| **Classical DQN** | MLP | Replay buffer (4000 transitions), target network (synced every 8 episodes), $\epsilon$-greedy ($\epsilon_0=0.9$, decay $0.965$) |

All classical networks use **manually-derived, exact analytic backpropagation** (no autograd library) — verified against finite differences to $1.63\times10^{-11}$ in `tests/test_gradients.py`. Gradient updates in every agent are **norm-clipped** (max norm 5.0) as a numerical-stability safeguard discovered necessary during development (an earlier configuration produced a rare NaN-softmax crash in classical PPO, since fixed).

---

## 6. Experiments and results

Seven experiments, run end-to-end and checkpointed at every boundary:

### 6.1 Main comparison (learning curves)
150 episodes × 3 seeds, $T=200$ steps/episode ($4.0\ \mathrm{s}$ flight). Final-20%-of-training return: QACLDN $-1295.72\pm128.58$, VQC-NE $-1312.11\pm120.89$, PPO $-1300.73\pm135.36$, SAC $-1299.76\pm134.62$, DQN $-1465.30\pm271.54$. The four actor-critic methods are statistically indistinguishable at this seed count (16.4-unit spread vs. ~70–78 s.e.m.); DQN is a clear, significant outlier.

### 6.2 Obstacle-density scalability
QACLDN vs. PPO, obstacle counts 2–12. Identical results at $n_{\text{obs}}=10,12$ due to hard-disk packing saturation of the rejection-sampling obstacle placer (not noise — the *realised* environment is literally identical at both settings once the 500-attempt placement budget saturates). QACLDN leads at 4 of 6 densities, most by $+48.35$ return units at $n_{\text{obs}}=8$.

### 6.3 Qubit-count sample complexity
$n\in\{2,4,6,8\}$, $L=2$. Non-monotonic final performance with an $n=4$ optimum ($-548.73$), consistent with the trainability/over-parameterisation trade-off reported in the variational-quantum-circuit ("barren plateau") literature, rather than a monotone theoretical scaling law.

### 6.4 $(n,L)$ performance map
$n\in\{4,6,8\}\times L\in\{1,2,3\}$, non-monotonic in both dimensions — no simple "more is better" trend at this budget.

### 6.5 NISQ noise resilience
Depolarising probability $p\in[0,0.01]$: collision rate stays within $[70.0,73.3]\%$ and mean return within $2.71$ units across two decades of $p$ — explained by the Bloch-contraction bound (§2) combined with the discrete $\arg\max$ policy's tolerance to small, correlated signal attenuation.

### 6.6 Wind-disturbance robustness
Classical PPO degrades monotonically with wind intensity ($20\%\to40\%$ collision rate); QACLDN's trace is non-monotone but not statistically resolvable at $N=20$ rollouts/point (binomial s.e. $\approx11.2$ points). The empirically-calibrated real wind level ($0.59g$) exceeds the entire tested sweep — an explicit, quantified coverage gap.

### 6.7 Final trajectory-quality evaluation
See README results table. QACLDN/VQC-NE trade higher collision rate for more active goal-seeking flight (PLR > 1), yielding the **best** mean returns; PPO/SAC converge to a conservative near-hover policy (PLR < 1, lowest collision rate) that never satisfies the dominant goal-attraction reward term, yielding the **worst** mean returns. Policy-gradient variance: QACLDN's parameter-shift estimator shows $3.3$–$4.5\times$ higher coordinate-wise variance than classical REINFORCE (consistent with the additional variance term $\mathrm{Var}[\hat\partial_iO]=\tfrac14(\mathrm{Var}[O_+]+\mathrm{Var}[O_-])$ intrinsic to the two-sided parameter-shift estimator), while both estimators closely track the ideal $1/B$ Monte-Carlo variance-reduction law.

---

## 7. Statistical scope and limitations

- **Reduced training budget.** The manuscript's target configuration (1200 episodes, $T=500$ steps) is reduced here to notebook-tractable scale (150 episodes, $T=200$ steps for the main comparison; shorter still for sweep experiments). This is stated wherever it materially affects statistical power, not hidden.
- **Reward rebalancing.** An initial reward configuration ($\lambda_c=4.0$, $\sigma_g=1.6\ \mathrm{m}$, $0.4\ \mathrm{m}$ success tolerance) caused the goal-attraction term to saturate near its floor for nearly the entire episode, flattening the learning signal. This was diagnosed empirically (by inspecting the resulting near-flat learning curves) and corrected to the values in §4 above.
- **Near-zero literal success rate.** Even after rebalancing, goal-reaching success (within $0.55\ \mathrm{m}$) remains ≈0% for all methods at this budget — a genuine sample-efficiency finding, cross-checked against independently-verified gradient code, not a bug.
- **A numerical stability bug was found and fixed during this study**: an unclipped gradient occasionally produced NaN action probabilities in classical PPO under the rebalanced (larger-magnitude) reward scale. Fixed with global gradient-norm clipping and NaN-safe softmax/dynamics guards (§5).
