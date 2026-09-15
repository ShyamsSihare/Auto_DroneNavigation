# Data — Provenance, Schema, and Licensing

**No raw data is vendored in this repository** (see `.gitignore`). This document tells you exactly what each dataset is, its exact schema, the exact statistics computed from it by `src/analysis.py`, and where to obtain it if you want to reproduce the calibration step.

If you already have the three archives, place them as:

```
data/raw/archive.zip                     # UAV Navigation Telemetry Log
data/raw/icmcis-drone-tracking.zip       # ICMCIS Multi-Sensor Drone-Detection Corpus
data/raw/gym-pybullet-drones-main.zip    # physical-parameter reference (not executed)
```

`src/analysis.py` unzips and reads these directly; no other setup is required.

---

## 1. UAV Navigation Telemetry Log

**File:** `uav_navigation_dataset.csv` (inside `archive.zip`)
**Size:** 5,000 rows × 15 columns, single-vehicle, 1 Hz, contiguous 4999 s (≈83.3 min) session, `2023-01-01 00:00:00` → `2023-01-01 01:23:19`.

| Field | dtype | Unit | Role |
|---|---|---|---|
| `timestamp` | datetime64 | — | Sample time, 1 Hz |
| `latitude`, `longitude` | float64 | ° | GPS position |
| `altitude` | float64 | m | Barometric/GPS altitude |
| `imu_acc_x/y/z` | float64 ×3 | m/s² | Body-frame linear acceleration |
| `imu_gyro_x/y/z` | float64 ×3 | °/s | Body-frame angular rate |
| `lidar_distance` | float64 | m | Nearest-obstacle range |
| `speed` | float64 | m/s | Ground-speed magnitude |
| `wind_speed` | float64 | m/s | Ambient wind magnitude |
| `battery_level` | float64 | % | Remaining charge |
| `obstacle_detected` | int64 {0,1} | — | Binary obstacle-in-range flag |

**Exact sample statistics** (computed directly from the raw CSV; $\hat\mu$ = sample mean, $\hat\sigma$ = sample std):

| Field | $\hat\mu$ | $\hat\sigma$ | min | max |
|---|---:|---:|---:|---:|
| latitude [°] | 37.774837 | 0.005793 | 37.764900 | 37.784894 |
| longitude [°] | −122.419570 | 0.005712 | −122.429399 | −122.409410 |
| altitude [m] | 275.618774 | 130.846418 | 50.070985 | 499.912315 |
| imu_acc_x [m/s²] | 0.046108 | 1.726837 | −2.998485 | 2.999549 |
| imu_acc_y [m/s²] | 0.052663 | 1.732448 | −2.999711 | 2.999406 |
| imu_acc_z [m/s²] | −0.052058 | 1.707324 | −2.999339 | 2.998393 |
| imu_gyro_x [°/s] | 0.221150 | 103.797185 | −179.998007 | 179.924174 |
| imu_gyro_y [°/s] | −1.147514 | 104.284254 | −179.935186 | 179.736727 |
| imu_gyro_z [°/s] | −0.555783 | 104.146932 | −179.993975 | 179.989973 |
| lidar_distance [m] | 49.779509 | 28.784503 | 0.510419 | 99.995717 |
| speed [m/s] | 17.603942 | 7.206767 | 5.009685 | 29.996033 |
| wind_speed [m/s] | 10.042630 | 5.768743 | 0.000169 | 19.998794 |
| battery_level [%] | 60.012946 | 23.070952 | 20.021610 | 99.975190 |
| obstacle_detected | 0.047200 | 0.212088 | 0 | 1 |

**Used for:** empirical fit of the wind Ornstein–Uhlenbeck turbulence model ($\hat\tau_w=0.63\ \mathrm{s}$, $\hat\sigma_w=5.769\ \mathrm{m/s}$), LiDAR/IMU sensor-noise floors, obstacle-detection-vs-range validation. Full derivation in `docs/METHODOLOGY.md` §1.1, §1.3.

---

## 2. ICMCIS Multi-Sensor Drone-Detection Corpus

Real flight-test logs from four co-located sensors — **ALVIRA** (phased-array radar), **ARCUS** (surveillance radar), **DIANA** (passive RF/direction-finding), **VENUS** (acoustic) — across a 14-scenario corpus split `train` (7 scenarios) / `test` (7 scenarios).

**This study uses 5 of the 14 scenarios, `train` split only, and only the ALVIRA and DIANA channels:**

| Scenario | ALVIRA rows | ARCUS rows | DIANA rows | VENUS rows |
|---|---:|---:|---:|---:|
| Scenario_1_1 | 628 | 18,109 | 628 | 803 |
| Scenario_1_3 | 413 | 53,906 | 425 | 569 |
| Scenario_1_4 | 437 | 95,208 | 477 | 613 |
| Scenario_2_1 | 1,012 | 235,564 | 873 | 1,422 |
| Scenario_Parrot_a | 450 | 99,443 | 316 | 137 |
| **Total (5 used)** | **2,940** | 502,230 | **2,719** | 3,544 |

Not used (listed for transparency): `train` scenarios `Scenario_1_2_b`, `Scenario_2_2`; the entire `test` split (`Scenario_1_2_a`, `Scenario_1_4_a`, `Scenario_3_1`, `Scenario_3_1_a`, `Scenario_3_3`, `Scenario_3_4_text`, `Scenario_Parrot_d`); the ARCUS and VENUS channels (present in every scenario but not ingested); and each scenario's ground-truth trajectory file (`2020-09-29_14-10-56_v2.csv`).

**Fields extracted:**

| Sensor | Fields used | Raw columns available | Retained / raw rows |
|---|---|---:|---|
| ALVIRA | `Track_Timestamp`, `TrackPosition_{Latitude,Longitude,Altitude}`, `TrackVelocity_Speed`, `Track_Score` | 40 | 1,311 / 2,940 (44.6%, filtered on non-null `TrackPosition_Altitude`) |
| DIANA | `TargetSignal_snr_dB`, `TargetSignal_bearing_deg`, `TargetSignal_range_m`, `TargetClassification_{score,type}` | 15 | 2,681 / 2,719 (98.6%, filtered on non-null `TargetSignal_snr_dB`) |

**Used for:** empirical fit of the one-way RF path-loss exponent ($\hat n_{\mathrm{pl}}=1.994$, $\hat A=64.83\ \mathrm{dB}$, over $R\in[20,870]\ \mathrm{m}$), and real-world radar track-speed ($\bar v=12.01\ \mathrm{m/s}$, $\hat\sigma_v=9.72\ \mathrm{m/s}$) / track-confidence statistics used in the sensor-validation figure. Full derivation in `docs/METHODOLOGY.md` §1.2.

---

## 3. gym-pybullet-drones (physical-parameter reference)

**Not a dataset — an open-source simulation framework**, used here only as a source of physically-realistic quadrotor parameters via its `cf2x.urdf` Crazyflie 2.X vehicle description. **The framework itself is not executed** as part of this pipeline; the rigid-body dynamics simulator in `src/analysis.py` is an independent, custom RK4 implementation of the manuscript's own Newton–Euler equations.

| Parameter | `cf2x.urdf` value | This study's (rescaled) value | Scale factor |
|---|---:|---:|---:|
| Mass $m$ | 0.027 kg | 0.891 kg | ×33 |
| Arm length $\ell$ | 0.0397 m | 0.20 m | ×5.04 |
| Thrust-to-weight $\lambda_{T/W}$ | 2.25 | 2.50 | ×1.11 |
| Max speed | 30 km/h | 10.8 km/h ($v_{\max}=3.0$ m/s) | ×0.36 |
| Thrust coeff. $k_f$ | $3.16\times10^{-10}$ | derived via $c_T=\lambda_{T/W}mg/4$ | — |
| Torque coeff. $k_m$ | $7.94\times10^{-12}$ | $c_Q=0.015\,c_T$ | — |

**Source:** Panerati, J., Zheng, H., Zhou, S., Xu, J., Prorok, A., & Schoellig, A. P. (2021). *Learning to Fly — a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control.* IEEE/RSJ IROS 2021. Repository license: check the upstream repository directly — its terms are **not** altered or overridden by this repository's MIT license.

**Used for:** deriving the closed-form hover command $u_{\mathrm{hover}}=1/\sqrt{\lambda_{T/W}}\approx0.632$ that centres the 16-action discretised control set. Full derivation in `docs/METHODOLOGY.md` §1.4.

---

## Licensing note

This repository's `LICENSE` (MIT) covers only the code, documentation, and derived tables/figures authored here. Each dataset above retains its own original license/terms of use, which you must independently review before redistributing raw data from any of the three sources.
