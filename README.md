# Certifiably Robust Reinforcement Learning (CRRL)

This repository contains the implementation of **Certified Policy
Optimization** for learning policies that satisfy **Spatio-Temporal
Reach and Escape Logic (STREL)** specifications under bounded
disturbances.

The method integrates:

-   Differentiable **STREL monitoring**
-   Neural network verification via **LiRPA**
-   **Certified policy optimization**

The goal is to learn policies that **maximize a certified lower bound on
specification robustness**, providing guarantees that the task is
satisfied for all disturbances within a prescribed uncertainty set.

------------------------------------------------------------------------

# Certified Policy Optimization

The policy is trained using a **certified robustness objective** that
combines nominal robustness with a **verified lower bound on worst-case
robustness**.

## Training Overview

### 1. Policy Initialization

A neural policy `πθ` is initialized randomly.

A **LiRPA wrapper** is built around the rollout module to enable **bound
propagation through the trajectory computation graph**.

------------------------------------------------------------------------

### 2. Curriculum Training

Training runs for **K iterations**.

At iteration `k`:

    λ = k / (K - 1)
    ε = λ * ε_max

The disturbance magnitude is gradually increased so that:

-   early training occurs under **nominal conditions**
-   the policy is progressively exposed to **larger disturbances**

------------------------------------------------------------------------

### 3. Certified Bound Computation

For each batch of initial states:

-   compute a **certified lower bound** on robustness using LiRPA
-   compute the **nominal robustness** under zero disturbance

These are combined into a **mixed robustness estimate**

    m = λ * ρ_cert + (1 - λ) * ρ_nom

This stabilizes training by gradually shifting optimization toward
certified robustness.

------------------------------------------------------------------------

### 4. Optimization

The certification loss is computed over the batch and the policy
parameters are updated via **gradient descent**.

------------------------------------------------------------------------

### 5. Two-Stage Certification

Training uses two verification methods:

#### Phase 1 --- CROWN-IBP (first 90% of training)

-   fast
-   coarse bounds

#### Phase 2 --- CROWN (final 10%)

-   slower
-   tighter bounds
-   improves certified robustness

------------------------------------------------------------------------

At convergence, the policy maximizes **certified robustness over the
disturbance set**.

------------------------------------------------------------------------

# Experiments

We evaluate CRRL on **reach--avoid navigation tasks** under two types of
disturbances:

-   **Wind disturbances** (action perturbations)
-   **Sensor noise** (observation perturbations)

The following baselines are compared:

  Method   Description
  -------- --------------------------------------------------
  PO       Policy Optimization with stochastic disturbances
  RARL     Robust Adversarial Reinforcement Learning
  CRRL     Proposed Certified RL
  PPO      Scalar-reward RL baseline

Performance is evaluated using:

-   STREL robustness
-   Certified lower bounds
-   Certified satisfaction rate
-   Statistical confidence bounds

------------------------------------------------------------------------

# Sensor Noise Results

## Certified Robustness vs Disturbance

![Certified robustness
sensor](images/compare_sensor_policies_crown_sweep.png)

This plot shows **certified robustness bounds** as the sensor noise
magnitude increases.

### Observations

-   **CRRL maintains positive certified robustness under larger
    disturbances**
-   Baselines degrade more quickly
-   CRRL provides **stronger verifiable guarantees**

------------------------------------------------------------------------

## Trajectory Comparison

![Sensor trajectories](images/compare_sensor_policies.png)

Each column shows trajectories produced by:

1.  PO\
2.  RARL\
3.  CRRL

under **sensor noise ε = 0.01**.

Each row corresponds to a different disturbance realization.

The right column shows **robustness distributions over 1000 initial
states**.

### Observations

-   All methods achieve high empirical success rates (\>95%)
-   **CRRL maintains larger safety margins**
-   Robustness distributions shift toward **higher positive values**

------------------------------------------------------------------------

## PPO Baseline (Scalar Reward)

![PPO robustness](images/sensor_scalar_reward_cert.png)

This plot shows STREL robustness and certified lower bounds for a PPO
policy trained with a **scalar reward**.

Although PPO achieves reasonable empirical performance:

-   its **certified robustness bounds are weak**
-   certification often fails

This highlights the importance of **training with STREL-aligned
objectives**.

------------------------------------------------------------------------

# Additional Experiments

## Generalization to Strong Wind Disturbances

Training disturbance:

    ε_train = 0.05

Evaluation disturbances:

-   `ε = 0.2` (4× training)
-   `ε = 0.5` (10× training)

------------------------------------------------------------------------

### Wind = 0.2

![Wind 0.2](images/compare_wind_policies_testwind0.2.png)

------------------------------------------------------------------------

### Wind = 0.5

![Wind 0.5](images/compare_wind_policies_testwind0.5.png)

Each column shows policies trained with:

-   PO
-   RARL
-   CRRL

The right column shows **robustness distributions over 1000 rollouts**.

### Observations

As disturbances increase:

-   **PO and RARL increasingly violate the safety specification**
-   trajectories deviate toward obstacles
-   goal reachability degrades

CRRL:

-   maintains **larger obstacle clearance**
-   preserves **higher robustness values**
-   retains a larger fraction of **specification-satisfying
    trajectories**

Even under disturbances **10× larger than those used during training**,
CRRL maintains significantly stronger robustness.

------------------------------------------------------------------------

# Numerical Results (Sensor Noise)

  Method        ε=0.02       ε=0.04       ε=0.06       ε=0.08
  ------------- ------------ ------------ ------------ ------------
  PO LB         +0.247       +0.234       +0.214       +0.181
  RARL LB       +0.230       +0.227       +0.223       +0.216
  **CRRL LB**   **+0.400**   **+0.343**   **+0.305**   **+0.251**

CRRL consistently achieves:

-   **higher certified robustness margins**
-   stronger guarantees

even when certified satisfaction rates are comparable.

------------------------------------------------------------------------

# Key Takeaways

CRRL learns policies that are:

-   **empirically robust**
-   **provably robust**
-   resilient to **disturbances beyond training conditions**

Optimizing **certified robustness bounds** leads to policies that
maintain safety guarantees under uncertainty.

------------------------------------------------------------------------

# Future Directions

Future work will focus on:

-   scaling to **longer horizons**
-   improving **verification efficiency**
-   extending to **multi-agent systems**
-   handling **probabilistic disturbances**
-   deploying on **real robotic platforms**
