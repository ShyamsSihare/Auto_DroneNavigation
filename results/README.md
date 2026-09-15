# Results

`QACLDN_experimental_results.pdf` is the complete multi-page figure set; `figures/*.png` are the same 6 figures individually. Full captions and scientific explanations (with every equation and every number) are in `manuscript/figure_captions_explanations.tex`.

| # | File | Contents |
|---|---|---|
| 1 | `figures/fig1_sensor_validation.png` | Real-data sensor validation: DIANA radar path-loss fit, ALVIRA track speed/confidence, UAV telemetry wind time-series + OU calibration, LiDAR/IMU noise floors, obstacle-detection-vs-range, DIANA classification confidence |
| 2 | `figures/fig2_learning_dynamics.png` | Main comparison learning curves: smoothed return, return-distribution violin plot, cross-seed stability, improvement-over-baseline, all 5 methods × 3 seeds |
| 3 | `figures/fig3_scalability_sample_complexity.png` | Obstacle-density scalability (QACLDN vs. PPO) + qubit-count sample complexity + VQC parameter-count scaling |
| 4 | `figures/fig4_map_robustness.png` | $(n,L)$ circuit-design performance map + NISQ depolarising-noise resilience + wind-disturbance robustness |
| 5 | `figures/fig5_trajectory_quality.png` | Final trajectory-quality bar charts (collision rate, path-length ratio, mean return), representative flight-path visualisation, energy-efficiency trade-off scatter |
| 6 | `figures/fig6_gradient_variance_summary.png` | Quantum vs. classical policy-gradient variance vs. batch size, manuscript-style summary table |

**All data in every panel is a real, computed output of `src/analysis.py`** — see `docs/METHODOLOGY.md` for the exact numbers behind each panel, and the "honest reporting" notes on statistical significance and known artefacts (e.g. the obstacle-count-10/12 saturation, the near-zero literal success rate).
