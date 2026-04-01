# Certifiably Robust Reinforcement Learning (CRRL)

This repository contains the implementation of **Certified Policy
Optimization** for learning policies that satisfy **Spatio-Temporal
Reach and Escape Logic (STREL)** specifications under bounded
disturbances.

The method combines:

-   Differentiable **STREL monitoring**
-   Neural network verification via **LiRPA**
-   **Certified policy optimization**

The objective is to learn policies that **maximize a certified lower
bound on specification robustness**, providing guarantees that the task
is satisfied for all disturbances within a prescribed uncertainty set.

------------------------------------------------------------------------

# Method Overview

CRRL integrates learning and verification by optimizing a **certified
robustness objective** rather than a standard reward.

The approach trains a neural policy while **propagating verification
bounds through the full trajectory computation graph**.

------------------------------------------------------------------------

# Certified Policy Optimization

The training algorithm optimizes a mixture of:

-   **nominal robustness**
-   **certified robustness bounds**

obtained via LiRPA verification.

## Training Procedure

### 1. Policy Initialization

A neural policy `πθ` is initialized randomly.

A **LiRPA wrapper** is built around the rollout module to enable **bound
propagation through the trajectory**.

------------------------------------------------------------------------

### 2. Curriculum Learning

Training runs for **K iterations**.

At iteration `k`:

    λ = k / (K − 1)
    ε = λ ε_max

The disturbance magnitude is gradually increased so that:

-   training begins under **nominal conditions**
-   the policy progressively encounters **larger disturbances**

------------------------------------------------------------------------

### 3. Certified Bound Computation

For each batch of initial states:

-   compute a **certified lower bound** on STREL robustness
-   compute **nominal robustness** with zero disturbance

These are combined into a mixed estimate

    m = λ ρ_cert + (1 − λ) ρ_nom

This stabilizes training and gradually shifts optimization toward
**certified robustness**.

------------------------------------------------------------------------

### 4. Optimization

The certification loss is evaluated over the batch and policy parameters
are updated using **gradient descent**.

------------------------------------------------------------------------

### 5. Two-Stage Certification

Training uses two verification methods:

#### Phase 1 --- CROWN-IBP (first 90%)

-   fast
-   coarse bounds

#### Phase 2 --- CROWN (final 10%)

-   slower
-   tighter bounds
-   improves certified guarantees

------------------------------------------------------------------------

# Experimental Evaluation

We evaluate CRRL on **reach--avoid navigation tasks** under two types of
disturbances:

-   **Wind disturbances** (action perturbations)
-   **Sensor noise** (observation perturbations)

The following baselines are compared:

  Method   Description
  -------- -------------------------------------------
  PO       Policy Optimization
  RARL     Robust Adversarial Reinforcement Learning
  CRRL     Proposed Certified RL
  PPO      Scalar-reward RL baseline

Metrics include:

-   STREL robustness
-   certified lower bounds
-   certified satisfaction rate
-   statistical confidence bounds

------------------------------------------------------------------------

# Wind Disturbance Results

## Certified Robustness vs Disturbance

![Certified robustness wind](images/compare_policies_crown_sweep.png)

This plot shows the **certified robustness lower bounds** for policies
trained with PO, RARL, and CRRL as the wind disturbance magnitude
increases.

### Observations

-   **CRRL consistently achieves the highest certified robustness**
-   CRRL maintains **positive certified bounds for larger disturbances**
-   PO and RARL degrade more rapidly

CRRL therefore provides **stronger verifiable guarantees** under
increasing disturbance levels.

------------------------------------------------------------------------

## Trajectory Comparison

![Wind trajectories](images/compare_policies.png)

Columns correspond to policies trained with:

1.  PO\
2.  RARL\
3.  CRRL

under **wind disturbance ε = 0.05**.

The right column shows **robustness distributions across 1000 initial
states**.

### Observations

-   All methods achieve good nominal performance
-   **CRRL produces trajectories with larger obstacle clearance**
-   robustness distributions shift toward **higher positive values**

------------------------------------------------------------------------

## Generalization to Strong Disturbances

Training disturbance:

    ε_train = 0.05

Evaluation disturbances:

-   **ε = 0.2** (4× training)
-   **ε = 0.5** (10× training)

### Wind = 0.2

![Wind 0.2](images/compare_wind_policies_testwind0.2.png)

### Wind = 0.5

![Wind 0.5](images/compare_wind_policies_testwind0.5.png)

As disturbance magnitude increases:

-   PO and RARL increasingly violate the specification
-   trajectories deviate toward obstacles
-   goal reachability deteriorates

CRRL instead:

-   maintains **safer trajectories**
-   preserves **larger robustness margins**
-   retains a larger fraction of **specification-satisfying
    trajectories**

Even under disturbances **10× larger than those used during training**,
CRRL maintains strong robustness.

------------------------------------------------------------------------

# Sensor Noise Results

## Certified Robustness vs Disturbance

![Certified robustness
sensor](images/compare_sensor_policies_crown_sweep.png)

CRRL maintains **positive certified robustness for larger sensor noise
levels**, while baselines degrade faster.

------------------------------------------------------------------------

## Trajectory Comparison

![Sensor trajectories](images/compare_sensor_policies.png)

Columns correspond to PO, RARL, and CRRL policies under **sensor noise ε
= 0.01**.

Observations:

-   empirical satisfaction remains high (\>95%)
-   **CRRL maintains larger safety margins**
-   robustness distributions shift toward **higher positive values**

------------------------------------------------------------------------

## PPO Baseline (Scalar Reward)

![PPO robustness](images/sensor_scalar_reward_cert.png)

Although PPO achieves reasonable empirical performance, its **certified
robustness bounds are weak**, highlighting the importance of
**STREL-aligned training objectives**.

------------------------------------------------------------------------

# Key Takeaways

CRRL learns policies that are:

-   **empirically robust**
-   **provably robust**
-   resilient to **disturbances beyond training conditions**

Optimizing **certified robustness bounds** leads to policies that
maintain safety guarantees under uncertainty.

------------------------------------------------------------------------

# Future Work

Future work will focus on:

-   scaling to **longer horizons**
-   improving **verification efficiency**
-   extending to **multi-agent systems**
-   handling **probabilistic disturbances**
-   deploying on **real robotic platforms**
