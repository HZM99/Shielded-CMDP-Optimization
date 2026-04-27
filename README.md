# Hybrid Shielded-CMDP Framework for Safe Smart Grid Control

**Author:** Hasan Muhaidat (supervised by Dr. Zekun Guo)  
**Affiliation:** University of Hull  
**Core Implementation:** Conservative Bounds Shielding for Distribution Network Voltage Control

## Executive Summary
This repository implements a **Hybrid Shielded-CMDP Framework**. The architecture combines First-Order Constrained Optimization in Policy Space (FOCOPS) with a deterministic safety shield. By projecting agent actions into conservative bounds (10% power margin), the framework achieves 100% constraint satisfaction in active distribution networks (ADNs).

## Scientific Contributions
* **Deterministic Safety Guarantee:** Introduces a structural safety mechanism that ensures constraint satisfaction via hard-clipping rather than standard Lagrangian penalty optimization.
* **Performance Gains:** Demonstrated a **7.60% increase in cumulative reward** and a **5.1% reduction in episodic cost** compared to unshielded baselines.
* **Positive Scaling:** Empirical evidence shows the shield's effectiveness increases with system complexity (1.92× improvement when scaling from IEEE 33-bus to 69-bus systems).
* **Real-time Feasibility:** Shielding logic adds negligible computational overhead (0.18 ms/action latency).

## Technical Architecture

```text
[Input: State] -> [FOCOPS Policy] -> [Unsafe Action] 
                                          |
                                          v
[Conservative Bounds Shield] <--- [Deterministic Projection]
                                          |
                                          v
[Result: 100% Safe Action] -> [Pandapower / IEEE Environment]
```

## Experimental Results
Results validated on IEEE 33-bus and 69-bus systems across 1M interactions (n=3 independent seeds).

| Metric | Baseline (FOCOPS) | Hybrid Shielded | Improvement | p-value |
|--------|-------------------|-----------------|-------------|---------|
| Cumulative Reward | -2381.67 ± 305.63 | -2200.74 ± 295.32 | +7.6% | 0.0489* |
| Episodic Cost | -4.05M ± 53k | -3.84M ± 28k | -5.1% | 0.0045† |
| Shield Latency | N/A | 0.18 ms | N/A | N/A |

(*) p < 0.05, (†) p < 0.01

## Repository Structure
* `/rl_constrained_smartgrid_control`: Core environment package (Gymnasium-compatible).
* `shield_model.py`: Implementation of the conservative bounds projection logic.
* `launch_focops_hybrid.py`: Main training script for the shielded agent.
* `Notebooks/`: Documentation of shield mechanics and statistical validation.
* `final_results/`: Trained model checkpoints and TensorBoard logs.

## Quick Start
```bash
# Clone and Install
git clone https://github.com/HZM99/Shielded-CMDP-Optimization.git
cd Shielded-CMDP-Optimization
conda env create -f omnisafe310_env.yml
pip install -e .

# Run Shielded Training (69-bus)
python launch_focops_hybrid.py --env-id IEEE69-Hybrid-v0 --epochs 100 --seed 0
```

## Citation
If you use this framework in your research, please cite:

```bibtex
@mastersthesis{muhaidat2025shield,
  author = {Muhaidat, Hasan},
  title = {Safe Reinforcement Learning for Smart Grid Control: Conservative Bounds Shielding},
  school = {University of Hull},
  year = {2025},
  note = {MSc Dissertation}
}
```

## Contact
Hasan Muhaidat  
h.muhaidat-2024@hull.ac.uk  
[ORCID: 0009-0009-0036-9581]
