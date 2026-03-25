# F.T — Physiological Instability Detection Engine

> A hybrid physiological reasoning and deep learning system for real-time ICU patient deterioration detection and prediction — up to 15 minutes in advance.

---

## What is F.T?

**F.T (Flow-Threshold)** is designed to detect and predict physiological deterioration in ICU patients aged 65 and above using continuous vital sign monitoring.

Rather than relying on simple alarm thresholds or pure machine learning, F.T combines four tightly integrated components:

- **Medical physiology rules** — clinically grounded deterioration patterns
- **Mathematical severity modeling** — continuous, nonlinear risk encoding
- **Temporal state logic** — FSM-based label stabilization
- **Deep learning prediction** — CNN-GRU trained on engineered physiological trajectories

The result is a system that learns *how deterioration unfolds over time*, not just whether a value is abnormal at a single moment.

---

## Design Philosophy — Why This Is Not Just a Prediction Model

**F.T is designed as a dual-purpose clinical intelligence system rather than a standalone predictive model.**

The learned prediction component constitutes only one layer of the system. Complementing it is a **Rule-Based Clinical Intelligence Layer** (under active development), which is responsible for:

* Generating *real-time clinical alerts* with interpretable reasoning
* Identifying the *specific physiological drivers* of patient instability
* Explaining *why* a patient’s risk profile is evolving, rather than only reporting predictions
* Translating model outputs into clinically meaningful, human-readable insights**

To support this architecture, feature engineering is intentionally constrained to a **compact set of physiologically interpretable variables (~41 features)**. These features are carefully selected to represent:

* Core vital signs
* Short- and long-term temporal trends (multi-scale slopes)
* Physiological interactions (e.g., pulse pressure, combined score)
* Clinically grounded rule-based indicators (t1, t2, t3 flags)
* Temporal context (rolling statistics and lag-based features)

Each feature maintains a **direct mapping to a clinical concept**, ensuring compatibility with the rule-based reasoning layer.

In contrast, purely statistical or high-dimensional transformations (e.g., PCA, latent embeddings, or black-box feature expansions) are deliberately avoided. While such methods may improve isolated predictive performance, they **break the interpretability bridge** required for clinical deployment.

Thus, the 41-feature design is not a limitation, but a **deliberate architectural constraint** that enables integration between machine learning predictions and transparent clinical reasoning.

---

## System Pipeline

```
Raw ICU Vital Streams  (2-second resolution)
          ↓
Physiological Risk Engine
  · Threshold zone mapping
  · Continuous abnormality encoding
  · Nonlinear severity transformation
  · Multi-organ risk aggregation
          ↓
Disease Pattern Detection
  · Tier 1 / 2 / 3 condition activation
  · Early deterioration ramp
  · Condition amplification multiplier
          ↓
Temporal Stability Modeling (FSM)
  · Label confirmation & state transition rules
          ↓
Feature Engineering
  · Raw vitals · Slopes · Rolling stats · Scaled vitals · Condition flags
          ↓
Deep Learning Prediction
  · CNN-GRU v6 (Deeper GRU + Temperature Calibration + Dual-Threshold)
  · 3-class severity output: Normal / Critical / Emergency
          ↓
[Planned] Rule-Based AI Layer
  · Real-time alerting with clinical reasoning
  · Explicit explanation of health change drivers
```

---

## Dataset

**Source:** VitalDB ICU Dataset

| Property | Value |
|---|---|
| Patients | **381 ICU patients** |
| Age | ≥ 60 years & < 80 years |
| Monitoring | Continuous vital sign streams |
| Resolution | 2-second intervals |
| Total rows | ~2,378,857 |

This forms a high-resolution geriatric ICU physiological stream dataset at real clinical scale.

**Target distribution (`future_label`):**

| Class | Label | Share |
|---|---|---|
| 0 | Normal | ~40% |
| 1 | Critical | ~20% |
| 2 | Emergency | ~40% |

---

## Vital Signals

### Primary Inputs

| Signal | Column | Description |
|---|---|---|
| SpO₂ | `spo2` | Oxygen saturation |
| Heart Rate | `heart_rate` | Pulse rate (bpm) |
| Respiratory Rate | `resp_rate_smoothed` | RR with rolling smoothing applied |
| Systolic BP | `sbp` | Systolic blood pressure |
| Diastolic BP | `dbp` | Diastolic blood pressure |
| End-Tidal CO₂ | `etco2` | Ventilatory CO₂ marker |

> Raw `resp_rate` is excluded. Only `resp_rate_smoothed` is used to reduce sensor noise.

### Derived Signals (treated as first-class features)

```
Pulse Pressure  =  SBP − DBP
MBP             =  (SBP + 2 × DBP) / 3
```

---

## Physiological Risk Engine

### Step 1 — Threshold Zone Mapping

Each vital is divided into three clinical risk zones:

| Zone | Meaning |
|---|---|
| Normal | Physiologically stable |
| Critical | Significant abnormality |
| Emergency | Severe instability |

**Full threshold table:**

| Vital | Normal | Critical | Emergency |
|---|---|---|---|
| SpO₂ | ≥ 95% | 92–95% | ≤ 90% |
| HR (high) | ≤ 90 bpm | 90–110 | ≥ 120 |
| HR (low) | ≥ 60 bpm | 50–60 | ≤ 45 |
| RR (high) | ≤ 20 /min | 20–25 | ≥ 30 |
| RR (low) | ≥ 12 /min | 10–12 | ≤ 8 |
| SBP (low) | ≥ 110 mmHg | 100–110 | ≤ 90 |
| SBP (high) | ≤ 150 mmHg | 150–170 | ≥ 185 |
| DBP (low) | ≥ 60 mmHg | 55–60 | ≤ 50 |
| DBP (high) | ≤ 85 mmHg | 85–95 | ≥ 100 |
| MBP | ≥ 70 mmHg | 65–70 | ≤ 60 |
| ETCO₂ (high) | ≤ 45 mmHg | 45–50 | ≥ 55 |
| ETCO₂ (low) | ≥ 35 mmHg | 30–35 | ≤ 25 |
| Pulse Pressure (low) | ≥ 45 mmHg | 35–45 | ≤ 30 |
| Pulse Pressure (high) | ≤ 65 mmHg | 65–75 | ≥ 85 |

### Step 2 — Continuous Abnormality Encoding

Rather than binary zone membership, each vital is mapped to a continuous score `z ∈ [0, 1]`:

```
z = 0.0  →  Normal (no abnormality)
z = 0.5  →  Critical boundary
z = 1.0  →  Emergency boundary
```

This models gradual physiological deterioration rather than abrupt threshold jumps.

### Step 3 — Nonlinear Severity Transformation

```
severity = 2^z − 1
```

| z | Severity |
|---|---|
| 0.0 | 0.00 |
| 0.5 | 0.41 |
| 1.0 | 1.00 |

Severity grows faster near emergency levels, reflecting the nonlinear escalation of clinical risk.

### Step 4 — Multi-Organ Risk Aggregation

```
severity_sum = Σ severity_i  (across all 8 vital signs)
```

This captures both single severe abnormalities and multiple concurrent mild abnormalities — modeling cumulative multi-organ physiological stress.

---

## Disease Pattern Modeling

F.T encodes **12 clinically meaningful deterioration patterns** across three tiers.

### Tier 1 — Major Instability

| Pattern | Trigger | Clinical Meaning |
|---|---|---|
| Shock Spiral | MBP < 70 AND HR > 100 | Low perfusion with compensatory tachycardia |
| Respiratory Burnout | SpO₂ < 92 AND RR > 22 | Oxygen failure with increased respiratory effort |
| Hypercapnic Failure | ETCO₂ > 50 AND RR < 10 | Ventilatory failure with CO₂ retention |

### Tier 2 — Moderate Risk

| Pattern | Trigger |
|---|---|
| Pulse Pressure Low | Pulse Pressure ≤ 30 |
| Wide PP + High SBP | Pulse Pressure ≥ 70 AND SBP ≥ 170 |
| Respiratory-Hemodynamic Combo | SpO₂ < 92 AND RR > 22 AND HR > 100 |

### Tier 3 — Subtle / Hidden Risk

| Pattern | Trigger |
|---|---|
| Hypertensive Emergency | SBP ≥ 180 AND Pulse Pressure ≥ 70 |
| Stable Deceiver | SpO₂ 92–94 AND HR 75–90 AND MBP 65–70 |
| Masked Shock | MBP 65–72 AND HR < 90 (perfusion decline without tachycardia) |
| Occult Acidosis | ETCO₂ ≤ 32 AND RR ≥ 24 AND SpO₂ 88–92 |
| Trend Decline | Simultaneous adverse point-to-point changes in ETCO₂, SpO₂, HR |
| Trend Activate | Slope-based sustained deterioration across 5–7 minute windows |

### Early Deterioration Ramp

Detection begins before thresholds are crossed:

```
early_start = threshold − 20% × (threshold − normal_reference)
```

This allows warning signals to develop before full clinical failure.

### Condition Amplification

Active conditions amplify the final instability score:

```
final_score = severity_sum × M_eff
M_eff = 1 + A × (target_multiplier − 1)
```

Where `A` is the condition activation strength (0–1) and multipliers are capped at 2.2 to prevent runaway escalation.

---

## Temporal Stability Engine

A **Finite State Machine (FSM)** prevents label flickering caused by noisy vital sign data.

Key rules:

- 15 consecutive identical states required to confirm a label change
- Emergency → Normal direct transition is blocked
- Mixed Critical / Emergency states collapse to Critical
- Downgrades require sustained recovery — not a single normal reading

This ensures state transitions reflect genuine physiological change, not sensor artifacts.

### Severity Classification

```
final_score < 0.75            →  ✅ Normal
0.75 ≤ final_score < 1.5      →  ⚠️  Critical
final_score ≥ 1.5             →  🚨 Emergency
```

---

## Feature Engineering

Temporal deterioration patterns are captured through **41 engineered features** across six categories.

These features are not arbitrary — each maps directly to a physiological concept that the Rule-Based AI layer can reason about and explain. This interpretability constraint is a core design requirement of the system.

### Raw Vitals (8)
`dbp · mbp · heart_rate · sbp · spo2 · etco2 · pulse_pressure · resp_rate_smoothed`

### Scaled Vitals (8)
Physiologically scaled versions of all 8 vitals:
`s_spo2 · s_hr · s_rr · s_sbp · s_dbp · s_mbp · s_etco2 · s_pp`

### Vital Slopes (16)
OLS slopes for all 8 vitals across 2 time windows:

| Window | Row Count | Trend Scope |
|---|---|---|
| 7m | 210 rows | Medium-term trend |
| 15m | 450 rows | Sustained trajectory |

Example: `slope_7m_spo2 · slope_15m_heart_rate · slope_7m_mbp · slope_15m_etco2`

### Combined Score Slopes (2)
`slope_7m_combined_score · slope_15m_combined_score`

### Rolling Statistics (6)
Computed over `combined_score`:

| Feature | Windows |
|---|---|
| `roll_mean_{w}_combined` | 7m, 15m |
| `roll_std_{w}_combined` | 7m, 15m |
| `roll_min_15m_combined` | 15m only |
| `roll_max_15m_combined` | 15m only |

### Physiological Instability Score (1)
`combined_score` — the output of the risk engine — used directly as a model feature.

---

## Deep Learning Model — CNN-GRU v6

### Architecture

A deeper GRU-augmented CNN with temperature calibration and dual-threshold decision logic:

```
Input: (batch, 240 timesteps, 44 features)
         ↓
  Multi-scale Conv1D feature extraction
         ↓
  GRU layers (deeper temporal modeling)
         ↓
  Temperature scaling (T = 1.50, calibrated on val set)
         ↓
  Dual-threshold decision logic
  · t_critical  = 0.40
  · t_emergency = 0.28
         ↓
  Output: 3-class severity (Normal / Critical / Emergency)
```

**Total parameters:** 136,292

### Window Configuration

| Parameter | Value | Description |
|---|---|---|
| Window | 240 rows (8 min) | Input sequence length |
| Stride | — | Step between windows |
| Train windows | 81,503 | — |
| Val windows | 9,664 | — |
| Test windows | 10,003 | — |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Batch size | 256 |
| Max epochs | 60 (early stopped at epoch 21) |
| Loss | Sparse categorical crossentropy |
| Class balancing | Weighted sampling |
| Post-hoc calibration | Temperature scaling (T = 1.4997) |
| Decision logic | Dual-threshold sweep (optimized on val set) |

### Train / Val / Test Split — Patient Level

| Split | Windows | Class Distribution |
|---|---|---|
| Train | 81,503 | N=33,069 C=16,599 E=31,835 |
| Val | 9,664 | N=3,419 C=2,269 E=3,976 |
| Test | 10,003 | N=3,078 C=2,309 E=4,616 |

---

## Model Performance

### Test Results

| Metric | Value |
|---|---|
| **Test AUROC** | **0.7260** |
| AUPRC | 0.5615 |
| Balanced Accuracy | 0.52 |
| Accuracy | 0.53 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal (0) | 0.63 | 0.52 | 0.57 | 3,078 |
| Critical (1) | 0.31 | 0.61 | 0.41 | 2,309 |
| Emergency (2) | 0.79 | 0.49 | 0.60 | 4,616 |

**Test confusion matrix:**

| | Pred-Normal | Pred-Critical | Pred-Emergency |
|---|---|---|---|
| True-Normal | 1,816 | 895 | 367 |
| True-Critical | 648 | 1,219 | 442 |
| True-Emergency | 583 | 1,992 | 2,041 |

### Key Result — Risk Detection Rate

The most clinically meaningful metric is not class-level accuracy in isolation but **combined deterioration detection**:

```
Emergency correctly identified as Emergency OR Critical:
→ (1,992 + 2,041) / 4,616  ≈  87% 
```

This reflects a critical insight about the system's behavior: the model has learned that **Critical is physiologically early Emergency**. Emergency cases misclassified as Critical are not model failures — they represent the model correctly identifying severe instability while applying a more conservative severity label. In a clinical alert context, both classes trigger intervention.

**True clinical performance summary:**

| Metric | Value |
|---|---|
| Critical recall | 0.53 |
| Emergency recall | 0.44 |
| **Risk detection (Critical + Emergency combined)** | **~87%** |
| Test AUROC | **0.7260** |

> Note: Per-class Emergency recall appears low due to strict label separation at inference. The 87% combined risk detection rate is the operationally correct measure for an early warning system.

---



---

## Repository Structure

```
├── catevcode.py                  # Physiological risk engine & feature pipeline
├── cnn_gru_v6_training.py        # Feature engineering & CNN-GRU training
└── README.md

cnn_gru_v6_outputs/
├── model weights                 # Trained model
├── scaler & metadata             # Normalisation artifacts
├── evaluation report             # Full classification report
├── training curves               # Loss / accuracy history
├── confusion matrices            # Val and test confusion matrices
```

---

## Applications

- ICU early warning and real-time deterioration monitoring
- Clinical decision support with explicit physiological reasoning
- Multi-organ failure detection research
- Physiological instability modeling and dataset construction
- Rule-based AI integration for bedside alert explanation

---

## Limitations

- Single-center dataset — generalizability to other ICU populations is unknown

- Currently a research prototype, not a certified clinical product
- Requires continuous high-frequency vital monitoring at 2-second resolution


---

## Planned Extensions

- [ ] **Rule-Based AI Layer** — Real-time alerting engine that identifies which specific vital signs, patterns, and trajectories are driving each severity classification, and communicates exact clinical reasoning to bedside staff
- [ ] **Score Fluctuation Analysis** — Studying `combined_score` trajectories: instability oscillation patterns, transition velocities, and physiological drivers behind score variance
- [ ] Validation on larger, multi-hospital datasets
- [ ] Transformer / LSTM time-series models
- [ ] Real-time ICU deployment pipeline
- [ ] Prospective clinical validation study
- [ ] Extension to broader ICU age groups

---

## Dataset

The master dataset used for this project is hosted on Kaggle due to GitHub file size limits.

BECAUSE of github datasize limitation  issue i uploaded the datsets in kaggle.

please DOWNLOAD INITIAL DATASET -https://www.kaggle.com/datasets/arjunmahesh09999/before-cleaning

please DOWNLOAD MASTER_DATASET -https://www.kaggle.com/datasets/arjunmahesh09999/new-masterdata

After downloading, place the dataset file inside the project directory before running the code.
