# Architecture — Code Map

`src/analysis.py` is written in [Jupytext "percent" format](https://jupytext.readthedocs.io/) (`# %%` cell markers), so it is simultaneously a plain runnable Python script and a losslessly-convertible Jupyter notebook (`notebooks/QACLDN_experimental_analysis.ipynb` is generated from it via `jupytext --to notebook`, then executed via `jupyter nbconvert --execute`).

The script is organised into 15 sequential sections. Sections 0–7 define reusable infrastructure (no side effects beyond a few numerical self-tests); Sections 8–15 run experiments and are the only parts that take significant wall-clock time.

```
SECTION 0   Imports, global config (CONFIG dataclass), plotting theme
SECTION 1   Real-data ingestion: UAV telemetry log + ICMCIS radar/RF corpus
            -> wind OU-process fit, radar path-loss fit, sensor noise floors

SECTION 2   Quadrotor rigid-body dynamics
            class QuadrotorDynamics      RK4 integrator, motor allocation matrix
            class OUWindProcess          discrete-time Ornstein-Uhlenbeck turbulence

SECTION 3   Quantum circuit simulator
            class VQC                    batched statevector sim + parameter-shift grad
            apply_gate_batch(), apply_cnot_batch(), apply_depolarising_batch()
            [self-test: parameter-shift vs. finite differences]

SECTION 4   Environment
            class ObstacleField          rejection-sampled cylindrical obstacles
            class DroneNavEnv            Gym-style reset()/step(), reward function

SECTION 5   Classical function approximation
            class MLP                    2-layer network, manual forward+backward
            [self-test: manual backprop vs. finite differences]

SECTION 6   RL agents
            class QuantumActorCriticAgent    QACLDN / VQC-NE (entangled flag)
            class ClassicalActorCriticAgent  PPO / Soft-AC (soft flag)
            class DQNAgent                   Classical DQN

SECTION 7   Training orchestration
            run_episode(), train_quantum_agent(), train_classical_ppo(),
            train_dqn(), eval_policy()

SECTION 8   Experiment 1 -- main comparison (learning curves)
SECTION 9   Experiment 2 -- obstacle-density scalability
SECTION 10  Experiment 3 -- qubit-count sample complexity
SECTION 11  Experiment 4 -- (n, L) performance map
SECTION 12  Experiment 5 -- NISQ noise resilience
SECTION 13  Experiment 6 -- wind-disturbance robustness
SECTION 14  Experiment 7 -- final trajectory-quality evaluation

SECTION 15  Figure generation (6 multi-panel figures -> multi-page PDF)
```

## Checkpointing

Every experiment (Sections 8–14) reads and writes a single `checkpoint.pkl` via `ckpt_load()` / `ckpt_save()` (defined just before Section 8). Each experiment block follows the pattern:

```python
if "expN" in CKPT:
    <reload results from CKPT["expN"]>
else:
    <run the experiment>
    CKPT["expN"] = {...}
    ckpt_save(CKPT)
```

Experiment 1 additionally checkpoints **per-seed** (`CKPT["exp1_partial"]`), so an interrupted multi-hour run resumes from the last completed seed rather than the last completed experiment. This is why re-running `python src/analysis.py` after any partial run is fast — only genuinely new work is executed.

## Data flow

```
uav_navigation_dataset.csv  ─┐
                              ├─► wind (τ_w, σ_w), sensor-noise floors ─► DroneNavEnv
ICMCIS ALVIRA/DIANA logs    ─┘                                            │
                                                                           ▼
gym-pybullet-drones cf2x.urdf ─► thrust-to-weight, mass/arm scaling ─► QuadrotorDynamics
                                                                           │
                                                                           ▼
                              VQC  ◄── policy/value backbone ──►  ClassicalActorCriticAgent / DQNAgent
                               │                                          │
                               └──────────────► train_* orchestration ◄───┘
                                                        │
                                                        ▼
                                          7 experiments -> CKPT -> 6 figures -> PDF
```

## Where to make common changes

| I want to... | Edit... |
|---|---|
| Change reward weights | `Config` dataclass, Section 0 |
| Change VQC architecture (qubits/layers/entanglement) | `Config.n_qubits_default`, `Config.L_layers_default`, or pass explicit args to `QuantumActorCriticAgent(...)` |
| Add a new classical baseline | Add a new agent class in Section 6, a `train_*` function in Section 7, and register it in `METHOD_NAMES` in Section 8 |
| Add a new figure | Append a new `# %% [markdown]` + figure-generation cell after Section 15's existing figures, call `pdf.savefig(fig)` before `pdf.close()` |
| Re-run with a larger training budget | Edit the relevant `n_episodes_*` / `n_seeds_*` fields in `Config`, delete `checkpoint.pkl`, re-run |
| Regenerate the executed notebook after any code change | `jupytext --to notebook src/analysis.py -o notebooks/QACLDN_experimental_analysis.ipynb && jupyter nbconvert --to notebook --execute --output notebooks/QACLDN_experimental_analysis.ipynb notebooks/QACLDN_experimental_analysis.ipynb` |
