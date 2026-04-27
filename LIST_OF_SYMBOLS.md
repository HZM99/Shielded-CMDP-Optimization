# List of Symbols

---

## General Notation

| Symbol | Description | Units/Type |
|--------|-------------|------------|
| $\mathbb{R}$ | Set of real numbers | - |
| $\mathbb{R}^n$ | n-dimensional Euclidean space | - |
| $\|\cdot\|$ | Absolute value or cardinality | - |
| $\|\|\cdot\|\|$ | Euclidean (L2) norm | - |
| $\mathbb{E}[\cdot]$ | Expected value | - |
| $\Pr(\cdot)$ | Probability | - |
| $\arg\min$ | Argument that minimizes | - |
| $\arg\max$ | Argument that maximizes | - |

---

## Reinforcement Learning

| Symbol | Description | Units/Type |
|--------|-------------|------------|
| $\mathcal{S}$ | State space | - |
| $s, s_t$ | State at time t | - |
| $\mathcal{A}$ | Action space | - |
| $a, a_t$ | Action at time t | - |
| $\pi$ | Policy (state-to-action mapping) | - |
| $\pi_\theta$ | Parameterized policy with parameters θ | - |
| $\theta$ | Neural network parameters | - |
| $r(s,a)$ | Reward function | - |
| $c(s,a)$ | Cost function (constraint violations) | - |
| $\gamma$ | Discount factor | $\in [0,1)$ |
| $V(s)$ | Value function (expected return from state s) | - |
| $V^R(s)$ | Reward value function | - |
| $V^C(s)$ | Cost value function | - |
| $Q(s,a)$ | Action-value function | - |
| $A(s,a)$ | Advantage function | - |
| $J(\pi)$ | Expected cumulative reward under policy π | - |
| $J_C(\pi)$ | Expected cumulative cost under policy π | - |
| $d$ | Constraint threshold (maximum allowed cost) | - |
| $\tau$ | Trajectory (sequence of states and actions) | - |
| $T$ | Episode length (time horizon) | timesteps |
| $\delta$ | KL divergence threshold (trust region) | - |
| $\lambda$ | GAE lambda parameter | $\in [0,1]$ |
| $D_{KL}$ | Kullback-Leibler divergence | - |

---

## Power Systems

| Symbol | Description | Units/Type |
|--------|-------------|------------|
| $V_i$ | Voltage magnitude at bus i | per unit (pu) |
| $V_{\text{nom}}$ | Nominal voltage (reference) | pu (typically 1.0) |
| $V_{\min}$ | Minimum allowed voltage | pu (typically 0.95) |
| $V_{\max}$ | Maximum allowed voltage | pu (typically 1.05) |
| $P_i$ | Active power injection at bus i | MW or kW |
| $Q_i$ | Reactive power injection at bus i | MVAr or kVAr |
| $P_{\text{load}}$ | Active power demand (load) | MW or kW |
| $Q_{\text{load}}$ | Reactive power demand (load) | MVAr or kVAr |
| $P_{\text{solar}}$ | Solar photovoltaic generation | MW or kW |
| $P_{ij}$ | Active power flow from bus i to bus j | MW or kW |
| $Q_{ij}$ | Reactive power flow from bus i to bus j | MVAr or kVAr |
| $R_{ij}$ | Resistance of line between buses i and j | Ω or pu |
| $X_{ij}$ | Reactance of line between buses i and j | Ω or pu |
| $Z_{ij}$ | Impedance of line between buses i and j | Ω or pu |
| $S_{\text{rated}}$ | Rated apparent power (inverter capacity) | MVA or kVA |
| $N_{\text{bus}}$ | Number of buses in the network | - |
| $n_a$ | Number of controllable DERs (action dimension) | - |
| $n_s$ | State space dimension | - |

---

## Conservative Bounds and Shielding

| Symbol | Description | Units/Type |
|--------|-------------|------------|
| $\epsilon$ (ε) | Conservative safety margin | pu (typically 0.02) |
| $\mathcal{C}_{\text{safe}}$ | Safe constraint set | - |
| $\mathcal{U}_P$ | Uncertainty set for active power | - |
| $\mathcal{U}_Q$ | Uncertainty set for reactive power | - |
| $\mathbf{Q}_{\text{raw}}$ | Raw action proposed by RL agent | kVAr vector |
| $\mathbf{Q}_{\text{safe}}$ | Safe action after shield projection | kVAr vector |
| $V_i^{\min}$ | Worst-case (minimum) voltage at bus i | pu |
| $V_i^{\max}$ | Worst-case (maximum) voltage at bus i | pu |
| $\mathbf{A}$ | Constraint matrix in QP formulation | - |
| $\mathbf{b}$ | Constraint vector in QP formulation | - |
| $\rho_{\text{intervene}}$ | Shield intervention rate | % |
| $\Delta Q_{\text{avg}}$ | Average shield intervention magnitude | kVAr |

---

## Experimental Metrics

| Symbol | Description | Units/Type |
|--------|-------------|------------|
| EpRet | Episodic return (cumulative reward) | - |
| EpCost | Episodic cost (cumulative violations) | - |
| EpLen | Episode length | timesteps |
| FPS | Training throughput (frames per second) | steps/second |
| CV | Coefficient of variation | % |
| $\mu$ | Mean (average) | - |
| $\sigma$ | Standard deviation | - |
| $p$ | p-value (statistical significance) | - |
| $d$ | Cohen's d (effect size) | - |
| $\alpha$ | Significance level | - (typically 0.05) |
| $n$ | Sample size (number of seeds) | - |

---

## Acronyms and Abbreviations

| Acronym | Full Name |
|---------|-----------|
| AC | Alternating Current |
| CMDP | Constrained Markov Decision Process |
| CPO | Constrained Policy Optimization |
| CV | Coefficient of Variation |
| DC | Direct Current |
| DER | Distributed Energy Resource |
| DMS | Distribution Management System |
| EV | Electric Vehicle |
| FOCOPS | First-Order Constrained Optimization in Policy Space |
| FPS | Frames Per Second |
| GAE | Generalized Advantage Estimation |
| IEEE | Institute of Electrical and Electronics Engineers |
| KL | Kullback-Leibler |
| MDP | Markov Decision Process |
| MPC | Model Predictive Control |
| NREL | National Renewable Energy Laboratory |
| OPF | Optimal Power Flow |
| PPO | Proximal Policy Optimization |
| PV | Photovoltaic |
| QP | Quadratic Programming |
| RL | Reinforcement Learning |
| RQ | Research Question |
| SCADA | Supervisory Control and Data Acquisition |
| SoC | State of Charge |

---
