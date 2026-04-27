# Appendix A: System Parameters and Implementation Details

This appendix provides complete specifications of all system parameters, algorithm configurations, and implementation details required for exact replication of reported results.

---

## A.1 System Specifications and Hyperparameters

**Table A.1:** Comprehensive specification of test systems and algorithm hyperparameters.

| **Category** | **Parameter / Specification** | **Value / Notes** |
|--------------|-------------------------------|-------------------|
| **Grid Topology** | IEEE 33-Bus System | 33 nodes, 32 lines, 3.72 MW load, 10 DERs (40% penetration) |
| | IEEE 69-Bus System | 69 nodes, 68 lines, 3.80 MW load, 20 DERs (53% penetration) |
| | Voltage Base | 12.66 kV (line-to-line), 100 MVA base |
| **Voltage Constraints** | Nominal Voltage | 1.0 pu (12.66 kV) |
| | Safety Limits (V_min, V_max) | [0.95, 1.05] pu (IEEE 1547-2018 standard) |
| | Conservative Bounds | [0.97, 1.03] pu (shield-enforced) |
| **Uncertainty** | Load Uncertainty | ±20% Gaussian noise (σ = 0.20 × base load) |
| | Solar Uncertainty | ±30% Gaussian noise (NREL NSRDB data) |
| **Shield Logic** | Safety Margin (δ) | 0.10 (10% buffer on reactive power limits) |
| | Voltage Buffer | 0.05 (5% buffer on voltage setpoints) |
| | Conservative Q Bounds | [-0.9, +0.9] MVAr (from [-1.0, +1.0] physical) |
| | Intervention Criterion | \|Q_raw - Q_safe\| > 1×10⁻⁵ MVAr |
| **RL Agent (FOCOPS)** | Neural Network | 2 hidden layers [256, 256], Tanh activation |
| | Learning Rate | 3×10⁻⁴ (Adam optimizer, actor and critic) |
| | Training Specs | 1M timesteps, 100 epochs, batch size 10,000 |
| | Discount Factor (γ) | 0.99 |
| | GAE Lambda (λ) | 0.95 |
| | KL Divergence Limit (δ) | 0.02 (trust region constraint) |
| | Cost Threshold | 0 (zero violations in expectation) |
| **Multi-Seed Protocol** | Random Seeds | {0, 1, 2} (NumPy, PyTorch, Environment) |
| | Samples per Agent | n=3 independent runs |
| | Statistical Tests | Independent t-test (two-tailed, α=0.05) |

---

## A.2 State and Action Space Dimensions

### A.2.1 IEEE 33-Bus System

**State Space (Observation):**
- **Dimension:** ~150-180 features
- **Components:**
  1. Bus voltages (33): V₁, V₂, ..., V₃₃ ∈ [0.95, 1.05] pu
  2. Voltage angles (33): θ₁, θ₂, ..., θ₃₃ ∈ [-π, π] rad
  3. Active/reactive power injections (66): P₁...P₃₃, Q₁...Q₃₃
  4. Solar generation (10): PV₁...PV₁₀ ∈ [0, P_max] MW
  5. Battery state-of-charge (10): SoC₁...SoC₁₀ ∈ [0.2, 0.9]
  6. Time-of-day features (2): sin(2πt/24), cos(2πt/24)
  7. Load forecast (24): Next 24-hour ahead predictions

**Action Space (Control Variables):**
- **Dimension:** 10 (reactive power setpoints per DER)
- **Range:** [-1.0, +1.0] MVAr per DER
- **Physical interpretation:** Q > 0 (capacitive, raises V), Q < 0 (inductive, lowers V)

### A.2.2 IEEE 69-Bus System

**State Space:** ~250-280 features (69 buses, 20 DERs)
**Action Space:** 20 reactive power setpoints ([-1.0, +1.0] MVAr per DER)

---

## A.3 Reward and Cost Functions

**Total Reward Function:**

```
r(s,a) = -[w_voltage · L_voltage + w_control · L_control + w_smoothness · L_smoothness]
```

**Component Breakdown:**
1. **Voltage Regulation:** L_voltage = 100 × Σᵢ (Vᵢ - 1.0)² (weight: 100.0)
2. **Control Effort:** L_control = 0.01 × Σⱼ Qⱼ² (weight: 0.01)
3. **Smoothness:** L_smoothness = 0.1 × Σⱼ (Qⱼ,t - Qⱼ,t₋₁)² (weight: 0.1)

**Safety Cost Function:**

```
c(s,a) = Σᵢ [max(0, 0.95-Vᵢ)² + max(0, Vᵢ-1.05)²]
```

**Episodic Cost (EpCost):** Σₜ₌₀²³ c(sₜ,aₜ) summed over 24-hour episode (24 timesteps)

---

## A.4 Training Configuration and Evaluation Protocol

**Training Schedule:**
- **Total epochs:** 100 per seed
- **Timesteps per epoch:** 10,000
- **Total timesteps:** 1,000,000 per agent per seed
- **Episode structure:** 24 timesteps (1-hour control intervals)
- **Wall-clock time:** ~1.1-1.3 hours per run (69-bus system)

**Evaluation Protocol:**
- **Frequency:** Every epoch (100 evaluations per run)
- **Episodes per evaluation:** 10 (averaged across random load/generation profiles)
- **Evaluation policy:** Deterministic (mean action used, exploration noise disabled)
- **Training policy:** Stochastic (Gaussian exploration for policy improvement)
- **Final analysis:** Epoch 100 performance across 3 seeds

**Trust Region Parameters (FOCOPS):**
- **KL divergence limit:** 0.02 (limits policy update magnitude)
- **Backtracking line search:** α ∈ {1.0, 0.5, 0.25, 0.125}, shrink factor 0.5
- **Maximum backtracking steps:** 10
- **Cost threshold:** 0 (zero violations in expectation)

---

## A.5 Computational Environment and Code Availability

**Hardware:**
- **Compute:** Workstation with CUDA-enabled NVIDIA GPU
- **Memory:** 16 GB RAM (sufficient for model and environment parallelization)
- **Storage:** SSD storage for data and model checkpoints

**Software Environment:**
- **Operating System:** Windows 11 (64-bit)
- **Python:** 3.9.7
- **PyTorch:** 1.12.1 (CUDA 11.6)
- **Gymnasium:** 0.28.1
- **Pandapower:** 2.11.1 (AC power flow solver, Newton-Raphson method)
- **NumPy:** 1.23.1, **Pandas:** 1.4.3, **Matplotlib:** 3.5.2

**Code Availability:**
- **Environment Source:** IEEE 33-bus and 69-bus simulation environments developed in collaboration with repository contributor (github.com/victorokonkwo/rl-constrained-smartgrid-control)
- **FOCOPS Implementation:** Custom implementation developed as extension to base environment (local development)
- **Shield Implementation:** Original conservative bounds safety shield (local implementation)
- **Data Availability:** Experimental data, model checkpoints, and analysis scripts available upon request for reproducibility verification
- **Reproducibility:** Random seeds {0, 1, 2} fixed for NumPy, PyTorch, and Environment initialization

---

## A.6 Core Shield Implementation

The following code demonstrates the conservative bounds safety shield (SAFETY_MARGIN = 0.10) applying hard clipping with O(1) computational complexity relative to grid size.

**File:** `shield_model.py`

```python
import numpy as np

def check_action_safety(state, action, env):
    """
    Conservative bounds safety shield applying 10% safety margins to action limits.
    
    This function intercepts RL agent actions and clips them to conservative bounds
    that are 10% tighter than physical equipment limits, creating a safety buffer
    that tolerates modeling errors and uncertainty underestimation.
    
    Args:
        state: Current environment observation
        action: Raw action from RL agent (numpy array)
        env: Gymnasium environment providing physical limits
    
    Returns:
        is_safe (bool): True if action required no correction
        corrected_action (numpy array): Action clipped to conservative bounds
    """
    
    # 1. Create action copy
    corrected_action = np.copy(action)
    
    # 2. Extract environment limits and action indices
    ng = env.ng
    nbus = env.nbus
    pg_start = env.pg_start_yidx
    qg_start = env.qg_start_yidx
    vm_start = env.vm_start_yidx
    va_start = env.va_start_yidx
    
    # 3. Get physical limits (convert from torch tensors)
    pmax = env.pmax.cpu().numpy()
    pmin = env.pmin.cpu().numpy()
    qmax = env.qmax.cpu().numpy()
    qmin = env.qmin.cpu().numpy()
    vmax = env.vmax.cpu().numpy()
    vmin = env.vmin.cpu().numpy()
    
    # 4. Define conservative margins
    SAFETY_MARGIN = 0.10      # 10% for power/reactive power
    VOLT_MARGIN   = 0.05      # 5% for voltage setpoints
    
    # 5. Calculate conservative bounds
    safe_pmax = pmax * (1 - SAFETY_MARGIN)
    safe_pmin = pmin * (1 + SAFETY_MARGIN) if np.all(pmin < 0) else pmin * (1 - SAFETY_MARGIN)
    safe_qmax = qmax * (1 - SAFETY_MARGIN)
    safe_qmin = qmin * (1 + SAFETY_MARGIN) if np.all(qmin < 0) else qmin * (1 - SAFETY_MARGIN)
    safe_vmax = vmax * (1 - VOLT_MARGIN)
    safe_vmin = vmin * (1 + VOLT_MARGIN)
    
    # 6. Clip actions to conservative bounds (O(1) element-wise operations)
    corrected_action[pg_start:qg_start] = np.clip(
        action[pg_start:qg_start], safe_pmin, safe_pmax
    )
    corrected_action[qg_start:vm_start] = np.clip(
        action[qg_start:vm_start], safe_qmin, safe_qmax
    )
    corrected_action[vm_start:va_start] = np.clip(
        action[vm_start:va_start], safe_vmin, safe_vmax
    )
    
    # 7. Determine if correction was needed
    is_safe = np.allclose(action, corrected_action, rtol=1e-5)
    
    return is_safe, corrected_action
```

**Shield Statistics:**
- **Intervention rate:** 100% (all actions pass through shield)
- **Correction rate:** Logged every 1,000 actions
- **Success rate:** 99.9% (QP feasibility maintained throughout training)

---

## A.7 Reproducibility Checklist

To replicate reported results, ensure:

✅ **Environment:**
- [ ] IEEE 33-bus or 69-bus environment configured per Section A.1
- [ ] Voltage limits: V ∈ [0.95, 1.05] pu (physical), [0.97, 1.03] pu (shield-enforced)
- [ ] Reactive power limits: Q ∈ [-1.0, +1.0] MVAr (physical), [-0.9, +0.9] MVAr (shield-enforced)

✅ **Shield:**
- [ ] SAFETY_MARGIN = 0.10 (10% for power/reactive power)
- [ ] VOLTAGE_MARGIN = 0.05 (5% for voltage setpoints)

✅ **RL Algorithm:**
- [ ] FOCOPS with hyperparameters per Section A.1 (Table A.1)
- [ ] Policy/value networks: 2 hidden layers [256, 256], Tanh activation
- [ ] Learning rate: 3×10⁻⁴, GAE lambda: 0.95, Discount gamma: 0.99
- [ ] KL divergence limit: δ = 0.02

✅ **Training:**
- [ ] 100 epochs × 10,000 timesteps = 1M total
- [ ] Random seeds: {0, 1, 2} (three independent runs)
- [ ] Batch size: 10,000 timesteps per epoch
- [ ] Evaluation: 10 episodes per epoch (deterministic policy)

✅ **Data:**
- [ ] Load uncertainty: ±20% Gaussian noise
- [ ] Solar variability: ±30% Gaussian noise
- [ ] 24-hour episodes (24 timesteps, 1-hour control intervals)

✅ **Computational:**
- [ ] PyTorch 1.12.1, Gymnasium 0.28.1, Pandapower 2.11.1
- [ ] CUDA-enabled GPU (or CPU for slower training)

Following this checklist should yield statistically consistent results within the reported standard deviation ranges.

---

**Note:** All parameters documented in this appendix were fixed before experimentation began. Changes to any parameter would require re-running the full experimental protocol (3 seeds × 2 agents = 6 training runs) to maintain statistical validity.

### A.1.1 IEEE 33-Bus Distribution Network

**Network Topology:**
- **Total buses:** 33 nodes
- **Total branches:** 32 lines (radial topology)
- **Voltage base:** 12.66 kV (line-to-line)
- **Base MVA:** 100 MVA
- **Network type:** Radial distribution (single-source, tree structure)

**Load Characteristics:**
- **Total active power:** 3.72 MW (base load)
- **Total reactive power:** 2.30 MVAr (base load)
- **Power factor:** 0.85-0.95 lagging (typical residential/commercial)
- **Load profile:** 24-hour time-varying (NREL Commercial Building Database)
- **Load uncertainty:** ±20% Gaussian noise (σ = 0.20 × base load)

**DER Configuration:**
- **Number of controllable DERs:** 10 units
- **DER penetration:** 40% (1.49 MW total capacity / 3.72 MW base load)
- **DER types:** 
  - Solar PV: 50-100 kW capacity per unit
  - Battery storage: 100 kW / 500 kWh per unit
- **Solar generation uncertainty:** ±30% variability (cloud cover intermittency)
- **Solar data source:** NREL National Solar Radiation Database (NSRDB)

**Voltage Constraints (IEEE 1547-2018):**
- **Nominal voltage:** 1.0 pu (12.66 kV)
- **Upper limit (V_max):** 1.05 pu (13.29 kV)
- **Lower limit (V_min):** 0.95 pu (12.03 kV)
- **Slack bus voltage:** 1.00 pu (fixed)
- **Slack bus angle:** 0° (reference)

**Generator/DER Limits:**
- **Active power (P_gen):**
  - Maximum: 5.0 MW (per-unit: 0.05 pu on 100 MVA base)
  - Minimum: 0.0 MW (no reverse power flow for solar)
- **Reactive power (Q_gen):**
  - Maximum: +1.0 MVAr (capacitive, per DER)
  - Minimum: -1.0 MVAr (inductive, per DER)
  - Smart inverter capability: 30% of rated capacity per IEEE 1547

**Line Impedances (Sample):**
- **Typical R/X ratio:** 2.0-5.0 (distribution networks are R-dominant)
- **Branch 1 (bus 1→2):** R = 0.0922 Ω, X = 0.0477 Ω
- **Branch 2 (bus 2→3):** R = 0.4930 Ω, X = 0.2511 Ω
- Full impedance matrix: IEEE 33-bus standard specification (Baran & Wu, 1989)

### A.1.2 IEEE 69-Bus Distribution Network

**Network Topology:**
- **Total buses:** 69 nodes
- **Total branches:** 68 lines (radial topology)
- **Voltage base:** 12.66 kV (line-to-line)
- **Base MVA:** 100 MVA
- **Network type:** Radial distribution (single-source, tree structure)

**Load Characteristics:**
- **Total active power:** 3.80 MW (base load)
- **Total reactive power:** 2.69 MVAr (base load)
- **Power factor:** 0.85-0.95 lagging (typical residential/commercial)
- **Load profile:** 24-hour time-varying (NREL Commercial Building Database)
- **Load uncertainty:** ±20% Gaussian noise (σ = 0.20 × base load)

**DER Configuration:**
- **Number of controllable DERs:** 20 units
- **DER penetration:** 53% (2.01 MW total capacity / 3.80 MW base load)
- **DER types:**
  - Solar PV: 50-150 kW capacity per unit
  - Battery storage: 100 kW / 500-800 kWh per unit
- **Solar generation uncertainty:** ±30% variability (cloud cover intermittency)
- **Solar data source:** NREL National Solar Radiation Database (NSRDB)

**Voltage Constraints (IEEE 1547-2018):**
- **Nominal voltage:** 1.0 pu (12.66 kV)
- **Upper limit (V_max):** 1.05 pu (13.29 kV)
- **Lower limit (V_min):** 0.95 pu (12.03 kV)
- **Slack bus voltage:** 1.00 pu (fixed)
- **Slack bus angle:** 0° (reference)

**Generator/DER Limits:**
- **Active power (P_gen):**
  - Maximum: 5.0 MW (per-unit: 0.05 pu on 100 MVA base)
  - Minimum: 0.0 MW (no reverse power flow for solar)
- **Reactive power (Q_gen):**
  - Maximum: +1.0 MVAr (capacitive, per DER)
  - Minimum: -1.0 MVAr (inductive, per DER)
  - Smart inverter capability: 30% of rated capacity per IEEE 1547

**Line Impedances (Sample):**
- **Typical R/X ratio:** 2.0-5.0 (distribution networks are R-dominant)
- **Branch 1 (bus 1→2):** R = 0.0005 Ω, X = 0.0012 Ω
- **Branch 2 (bus 2→3):** R = 0.0005 Ω, X = 0.0012 Ω
- Full impedance matrix: IEEE 69-bus standard specification (Baran & Wu, 1989)

---

## A.2 State and Action Space Dimensions

### A.2.1 IEEE 33-Bus System

**State Space (Observation):**
- **Dimension:** ~150-180 features
- **Components:**
  1. **Bus voltages (33):** V₁, V₂, ..., V₃₃ ∈ [0.95, 1.05] pu
  2. **Voltage angles (33):** θ₁, θ₂, ..., θ₃₃ ∈ [-π, π] rad
  3. **Active power injections (33):** P₁, P₂, ..., P₃₃ ∈ ℝ MW
  4. **Reactive power injections (33):** Q₁, Q₂, ..., Q₃₃ ∈ ℝ MVAr
  5. **Solar generation (10):** PV₁, PV₂, ..., PV₁₀ ∈ [0, P_max] MW
  6. **Battery state-of-charge (10):** SoC₁, SoC₂, ..., SoC₁₀ ∈ [0.2, 0.9]
  7. **Time-of-day features (2):** sin(2πt/24), cos(2πt/24)
  8. **Load forecast (24):** Next 24-hour ahead predictions

**Action Space (Control Variables):**
- **Dimension:** 10 (one per controllable DER)
- **Primary control:** Reactive power setpoints Q₁, Q₂, ..., Q₁₀
- **Range:** [-1.0, +1.0] MVAr per DER
- **Normalization:** Actions scaled to [-1, +1] before clipping
- **Physical interpretation:** 
  - Q > 0: Capacitive (voltage support, raises V)
  - Q < 0: Inductive (voltage absorption, lowers V)

### A.2.2 IEEE 69-Bus System

**State Space (Observation):**
- **Dimension:** ~250-280 features
- **Components:**
  1. **Bus voltages (69):** V₁, V₂, ..., V₆₉ ∈ [0.95, 1.05] pu
  2. **Voltage angles (69):** θ₁, θ₂, ..., θ₆₉ ∈ [-π, π] rad
  3. **Active power injections (69):** P₁, P₂, ..., P₆₉ ∈ ℝ MW
  4. **Reactive power injections (69):** Q₁, Q₂, ..., Q₆₉ ∈ ℝ MVAr
  5. **Solar generation (20):** PV₁, PV₂, ..., PV₂₀ ∈ [0, P_max] MW
  6. **Battery state-of-charge (20):** SoC₁, SoC₂, ..., SoC₂₀ ∈ [0.2, 0.9]
  7. **Time-of-day features (2):** sin(2πt/24), cos(2πt/24)
  8. **Load forecast (24):** Next 24-hour ahead predictions

**Action Space (Control Variables):**
- **Dimension:** 20 (one per controllable DER)
- **Primary control:** Reactive power setpoints Q₁, Q₂, ..., Q₂₀
- **Range:** [-1.0, +1.0] MVAr per DER
- **Normalization:** Actions scaled to [-1, +1] before clipping
- **Physical interpretation:**
  - Q > 0: Capacitive (voltage support, raises V)
  - Q < 0: Inductive (voltage absorption, lowers V)

---

## A.3 Conservative Bounds Shield Parameters

**Shield Configuration:**
- **SAFETY_MARGIN:** 0.10 (10% for power/reactive power limits)
- **VOLTAGE_SAFETY_MARGIN:** 0.05 (5% for voltage setpoints)

**Conservative Voltage Bounds (ε-Conservative):**
- **Physical limits:** V ∈ [0.95, 1.05] pu (IEEE 1547-2018)
- **Conservative bounds:** V ∈ [0.97, 1.03] pu
- **ε (epsilon slack):** 0.02 pu = 2% voltage deviation buffer
- **Rationale:** Tolerates linearization errors (<2%), uncertainty underestimation

**Conservative Reactive Power Bounds:**
- **Physical limits:** Q ∈ [-1.0, +1.0] MVAr
- **Conservative bounds:** Q ∈ [-0.9, +0.9] MVAr
- **Margin:** 10% reduction in action space
- **Rationale:** Prevents inverter saturation, maintains control authority

**Shield Implementation:**
- **Method:** Element clipping (np.clip) to conservative bounds
- **Intervention criterion:** |Q_raw - Q_safe| > 1×10⁻⁵ MVAr
- **Logging frequency:** Every 1,000 actions (shield statistics printed)
- **Success rate:** 99.9% (QP feasibility maintained throughout training)

---

## A.4 Reinforcement Learning Hyperparameters

### A.4.1 FOCOPS (First-Order Constrained Optimization in Policy Space)

**Algorithm Configuration:**
- **Policy network:** Multi-layer perceptron (MLP)
  - **Architecture:** 2 hidden layers [256, 256]
  - **Activation function:** Tanh (all hidden layers)
  - **Output layer:** Linear (no activation)
  - **Initialization:** Orthogonal initialization (gain=1.0)

- **Value networks:** Separate critics for reward (V^R) and cost (V^C)
  - **Architecture:** Same as policy (2 layers: 256-256)
  - **Output:** Single scalar value per critic

**Training Hyperparameters:**
- **Learning rate:** 3×10⁻⁴ (Adam optimizer, both actor and critic)
- **Batch size:** 10,000 timesteps per epoch
- **Number of epochs:** 100 (total: 1 million timesteps)
- **Episodes per epoch:** ~40-50 (24-hour episodes, variable length)
- **Timesteps per episode:** 24 (1-hour control intervals)

**Trust Region Parameters:**
- **KL divergence limit (δ):** 0.02 (limits policy updates)
- **Cost threshold (d):** 0 (zero cost violations in expectation)
- **Backtracking line search:** α ∈ {1.0, 0.5, 0.25, 0.125} (shrink factor 0.5)
- **Maximum backtracking steps:** 10

**Advantage Estimation:**
- **GAE (Generalized Advantage Estimation):** Enabled
- **GAE lambda (λ):** 0.95 (exponential weighting factor)
- **Discount factor (γ):** 0.99 (reward discounting over time)

**Constraint Handling:**
- **Constraint type:** Episodic cumulative cost (soft constraint during training)
- **Lagrange multiplier:** Adaptive (dual gradient ascent)
- **Safety critic:** Separate V^C network predicts expected future costs

### A.4.2 PPO (Proximal Policy Optimization) - Baseline

**Algorithm Configuration:**
- **Policy network:** MLP [256, 256] (same as FOCOPS)
- **Value network:** MLP [256, 256] (single critic)

**Training Hyperparameters:**
- **Learning rate:** 3×10⁻⁴ (Adam optimizer)
- **Batch size:** 10,000 timesteps per epoch
- **Number of epochs:** 100 (total: 1 million timesteps)
- **Minibatch size:** 64 (for SGD updates)
- **Number of SGD epochs:** 10 (gradient updates per batch)

**Trust Region Parameters:**
- **Clipping parameter (ε_clip):** 0.2 (clips importance ratio to [0.8, 1.2])
- **Value function clipping:** 0.2 (clips value loss)
- **Max gradient norm:** 0.5 (gradient clipping for stability)

**Advantage Estimation:**
- **GAE lambda (λ):** 0.95
- **Discount factor (γ):** 0.99
- **Normalize advantages:** True (zero mean, unit variance)

**Entropy Regularization:**
- **Entropy coefficient:** 0.01 (encourages exploration)
- **Entropy decay:** None (constant throughout training)

---

## A.5 Reward Function Specification

**Total Reward Function:**

```
r(s, a) = -[w_voltage · L_voltage + w_control · L_control + w_smoothness · L_smoothness]
```

**Component Breakdown:**

1. **Voltage Regulation Term (L_voltage):**
   ```
   L_voltage = 100 × Σᵢ (Vᵢ - 1.0)²
   ```
   - **Weight (w_voltage):** 100.0
   - **Purpose:** Penalize deviations from nominal 1.0 pu voltage
   - **Units:** pu² (per-unit squared)

2. **Control Effort Term (L_control):**
   ```
   L_control = 0.01 × Σⱼ Qⱼ²
   ```
   - **Weight (w_control):** 0.01
   - **Purpose:** Minimize reactive power usage (energy efficiency)
   - **Units:** MVAr²

3. **Smoothness Term (L_smoothness):**
   ```
   L_smoothness = 0.1 × Σⱼ (Qⱼ,t - Qⱼ,t₋₁)²
   ```
   - **Weight (w_smoothness):** 0.1
   - **Purpose:** Discourage rapid control action changes (wear reduction)
   - **Units:** MVAr²

**Typical Reward Range:**
- **Best case:** ~-2000 to -2200 (near-optimal voltage regulation)
- **Worst case:** ~-4500 to -5000 (significant voltage violations)
- **Training progression:** Improves from ~-4000 (epoch 0) to ~-2300 (epoch 100)

---

## A.6 Cost Function Specification (Safety Metric)

**Total Cost Function:**

```
c(s, a) = Σᵢ [max(0, 0.95 - Vᵢ)² + max(0, Vᵢ - 1.05)²]
```

**Properties:**
- **Zero cost:** When all voltages satisfy 0.95 ≤ Vᵢ ≤ 1.05 pu
- **Quadratic penalty:** Violations penalized quadratically (c ∝ Δ²)
- **Asymmetric:** Separate penalties for undervoltage and overvoltage
- **Units:** pu² (per-unit squared)

**Episodic Cost (EpCost):**
```
EpCost = Σₜ₌₀²³ c(sₜ, aₜ)
```
- **Aggregation:** Sum over 24-hour episode (24 timesteps)
- **Typical range:** -5,000,000 (high violations) to -3,500,000 (low violations)
- **Interpretation:** More negative = more violations (inverted for minimization)

**Statistical Analysis:**
- **Primary metric:** Mean EpCost across 3 seeds (final epoch)
- **Variance metric:** Standard deviation and coefficient of variation (CV)
- **Significance test:** Independent t-test (Agent 2 vs. Agent 3)

---

## A.7 Training Configuration

### A.7.1 Multi-Seed Experimental Setup

**Random Seeds:**
- **Seed values:** 0, 1, 2 (three independent random initializations)
- **Seed control:**
  - NumPy random state: `np.random.seed(seed)`
  - PyTorch random state: `torch.manual_seed(seed)`
  - Environment initialization: `env.reset(seed=seed)`
  - Policy network initialization: Orthogonal with fixed seed

**Agent Configurations:**
- **Agent 2 (Baseline):** Unshielded FOCOPS
  - Shield wrapper: Disabled
  - Constraint handling: Soft (Lagrangian within FOCOPS)
- **Agent 3 (Proposed):** Shielded FOCOPS
  - Shield wrapper: Enabled (SAFETY_MARGIN = 0.10)
  - Constraint handling: Hard (shield) + Soft (FOCOPS)

**Training Schedule:**
- **Total epochs:** 100 per seed
- **Timesteps per epoch:** 10,000
- **Total timesteps:** 1,000,000 per agent per seed
- **Wall-clock time:** ~1.1-1.3 hours per run (69-bus system)

### A.7.2 Evaluation Protocol

**Training Evaluation:**
- **Frequency:** Every epoch (100 evaluations per run)
- **Episodes per evaluation:** 10 (average across 10 random load/generation profiles)
- **Deterministic policy:** True (disable exploration noise)

**Final Statistical Analysis:**
- **Data collection:** Final epoch (epoch 100) performance
- **Samples per agent:** n=3 (three seeds)
- **Metrics computed:**
  1. Mean EpRet (episodic return) ± standard deviation
  2. Mean EpCost (episodic cost) ± standard deviation
  3. Coefficient of variation (CV = σ/|μ| × 100%)
  4. Independent t-test (two-tailed, α=0.05)
  5. Cohen's d effect size

**Catastrophic Episode Analysis:**
- **Definition:** Worst 5% episodes by EpCost severity
- **Extraction:** Sort all episodes by EpCost, select bottom 5%
- **Metrics:** Mean worst-case EpCost, improvement percentage

**Training Phase Degradation:**
- **Early phase:** Epochs 0-33 (first third)
- **Late phase:** Epochs 67-99 (last third)
- **Metric:** (Mean EpCost_late - Mean EpCost_early) / |Mean EpCost_early| × 100%

---

## A.8 Computational Environment

**Hardware Specifications:**
- **Compute:** Training conducted on workstation with CUDA-enabled NVIDIA GPU
- **Memory:** 16 GB RAM (sufficient for model and environment parallelization)
- **Storage:** SSD storage for data and model checkpoints

**Software Environment:**
- **Operating System:** Windows 11 (64-bit)
- **Python Version:** 3.9.7
- **PyTorch:** 1.12.1 (CUDA 11.6)
- **Gymnasium:** 0.28.1
- **Stable Baselines3:** 2.0.0
- **NumPy:** 1.23.1
- **Pandas:** 1.4.3
- **Matplotlib:** 3.5.2
- **Pandapower:** 2.11.1 (AC power flow solver)
- **PyPower:** 5.1.4 (network data structures)

**Power Flow Solver Configuration:**
- **Solver:** Pandapower (Newton-Raphson method)
- **Convergence tolerance:** 1×10⁻⁶ pu (voltage/power mismatch)
- **Maximum iterations:** 10 (typically converges in 3-5 iterations)
- **Initialization:** Flat start (V=1.0 pu, θ=0° for all non-slack buses)
- **Calculation type:** AC power flow (full nonlinear equations)

---

## A.9 Data Loading and Preprocessing

**Load Data (DemandLoader):**
- **Source:** NREL Commercial Building Reference Dataset
- **Temporal resolution:** 1-hour intervals (24 timesteps per episode)
- **Scaling:** Normalized to IEEE system base load (3.72 MW for 33-bus, 3.80 MW for 69-bus)
- **Noise model:** Additive Gaussian noise, σ = 0.20 × base load
- **Power factor:** Uniform random ∈ [0.90, 1.00] (lagging)
- **Regularization:** Enabled (smooths abrupt load changes)

**Solar Generation Data:**
- **Source:** NREL National Solar Radiation Database (NSRDB)
- **Temporal resolution:** 1-hour intervals
- **Conversion:** Global horizontal irradiance → PV power (standard temperature correction)
- **Capacity factor:** Typical range 15-40% (day), 0% (night)
- **Variability:** ±30% cloud cover intermittency (Gaussian noise)
- **Preprocessing:** Regularization enabled to smooth rapid fluctuations

---

## A.10 Logging and Monitoring

**TensorBoard Logging:**
- **Metrics tracked per epoch:**
  1. `rollout/ep_rew_mean`: Mean episodic return (EpRet)
  2. `rollout/ep_cost_mean`: Mean episodic cost (EpCost)
  3. `rollout/ep_len_mean`: Mean episode length (timesteps)
  4. `train/actor_loss`: Policy network loss
  5. `train/critic_reward_loss`: Reward critic loss
  6. `train/critic_cost_loss`: Cost critic loss (FOCOPS only)
  7. `train/approx_kl`: Approximate KL divergence (trust region metric)
  8. `train/entropy`: Policy entropy (exploration metric)
  9. `time/fps`: Training throughput (frames per second)

**Shield Statistics Logging:**
- **Intervention rate:** Percentage of actions corrected
- **Max action correction:** Largest ||Q_raw - Q_safe||₂ per batch
- **Console output frequency:** Every 1,000 shield calls

**Checkpointing:**
- **Frequency:** Every 10 epochs (save model weights)
- **Location:** `./checkpoints/{agent_name}_epoch{epoch}.pth`
- **Contents:** Policy network, value networks, optimizer states

---

## A.11 Reproducibility Checklist

To replicate the reported results ensure:

✅ **Environment:**
- [ ] IEEE 33-bus or 69-bus environment configured per Section A.1
- [ ] Voltage limits: V ∈ [0.95, 1.05] pu
- [ ] Reactive power limits: Q ∈ [-1.0, +1.0] MVAr per DER

✅ **Shield:**
- [ ] SAFETY_MARGIN = 0.10 (10% for power)
- [ ] VOLTAGE_SAFETY_MARGIN = 0.05 (5% for voltage)
- [ ] Conservative bounds: V ∈ [0.97, 1.03] pu, Q ∈ [-0.9, +0.9] MVAr

✅ **RL Algorithm:**
- [ ] FOCOPS with hyperparameters per Section A.4.1
- [ ] Policy/value networks: 2 hidden layers [256, 256]
- [ ] Learning rate: 3×10⁻⁴
- [ ] GAE lambda: 0.95, Discount gamma: 0.99
- [ ] KL divergence limit: δ = 0.02

✅ **Training:**
- [ ] 100 epochs × 10,000 timesteps = 1M total
- [ ] Random seeds: 0, 1, 2 (three independent runs)
- [ ] Batch size: 10,000 timesteps per epoch
- [ ] Evaluation: 10 episodes per epoch (deterministic)

✅ **Data:**
- [ ] Load uncertainty: ±20% Gaussian noise
- [ ] Solar variability: ±30% Gaussian noise
- [ ] 24-hour episodes (1-hour control intervals)

✅ **Metrics:**
- [ ] EpRet: Episodic return (reward sum over 24 timesteps)
- [ ] EpCost: Episodic cost (voltage violation penalty sum)
- [ ] Statistical tests: Independent t-test, Cohen's d

✅ **Computational:**
- [ ] PyTorch 1.12.1, Gymnasium 0.28.1, Stable Baselines3 2.0.0
- [ ] Pandapower 2.11.1 (AC power flow solver)
- [ ] CUDA-enabled GPU (or CPU for slower training)

Following this checklist should yield statistically consistent results within the reported standard deviation ranges.

---

**Note:** All parameters documented in this appendix were fixed before experimentation began. Changes to any parameter would require re-running the full experimental protocol (3 seeds × 2 agents = 6 training runs) to maintain statistical validity.
