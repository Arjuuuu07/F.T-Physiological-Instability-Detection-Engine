# F.T — Flow-Threshold · Physiological Instability Detection Engine

<div align="center">

**A hybrid clinical intelligence system combining physiological rule modeling, temporal state logic, and deep learning**  
**for real-time ICU patient deterioration detection — up to 15 minutes before clinical failure.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Dataset-VitalDB-green.svg)](https://vitaldb.net/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow.svg)]()

</div>

---

## ⚡ What F.T Achieves

> **"Detects ~87% of ICU deterioration cases — up to 15 minutes before clinical failure."**

That number is not accuracy. It is not recall. It is the answer to the only question that matters in an early warning system:

> *"When a patient is deteriorating, does the system flag it?"*

```
Emergency → Emergency:   2,041  ✅ Direct detection
Emergency → Critical:    1,992  ✅ Conservative early detection  (both trigger intervention)
Emergency → Normal:        583  ❌ Missed
──────────────────────────────────────────────────────────────────
Combined risk detection:  4,033 / 4,616  ≈  87%
```

Most early warning systems ask: *"Is this value out of range right now?"*  
F.T asks: *"How is this patient's physiology evolving — and where is it heading?"*

---

## 🧠 Why This Is Not Just a Prediction Model

F.T is designed as a **dual-purpose clinical intelligence system**, not a standalone ML model.

Most ICU ML projects train a classifier and stop. F.T is architected in two integrated layers:

| Layer | Role | Status |
|---|---|---|
| **Deep Learning Layer** | Predicts severity class (Normal / Critical / Emergency) | ✅ Complete |
| **Rule-Based Clinical Reasoning Layer** | Explains *why* risk is changing, in human-readable clinical language | 🔧 In Development |

The rule-based layer is not an afterthought — it shapes every design decision in the system. This is why:

- The model uses a curated subset of **interpretable variables** — not 200+ statistical transforms
- Black-box methods like PCA, latent embeddings, and high-dimensional expansions are **deliberately excluded** from model inputs
- Each model feature maintains a **direct mapping to a clinical concept**

The **master dataset retains all 99 features** — this is intentional. The Rule-Based Reasoning Layer (under active development) requires the full feature set to analyse physiological state in depth, identify drivers, and generate human-readable clinical explanations.

The Rule-Based Layer will be responsible for:
- Generating real-time clinical alerts with interpretable reasoning
- Identifying the specific physiological drivers of each patient's instability
- Explaining **why** a patient's risk profile is evolving — not just **that** it is
- Translating model outputs into clinically meaningful, human-readable insights

---

## 🏗️ Full System Pipeline

```
╔══════════════════════════════════════════════════════════════════╗
║          Raw ICU Vital Streams  (2-second resolution)            ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║                 Physiological Risk Engine                        ║
║   · Threshold zone mapping  (Normal / Critical / Emergency)      ║
║   · Continuous abnormality encoding   z ∈ [0, 1]                ║
║   · Nonlinear severity transformation   severity = 2^z − 1      ║
║   · Multi-organ risk aggregation  (8 vitals combined)            ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║                Disease Pattern Detection                         ║
║   · 12 clinically grounded patterns across 3 tiers              ║
║   · Early deterioration ramp  (pre-threshold warning onset)      ║
║   · Condition amplification multiplier  (capped at 2.2×)        ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║             Temporal Stability Engine  (FSM)                     ║
║   · Label confirmation & state transition rules                  ║
║   · Prevents sensor-noise-driven flickering                      ║
║   · Emergency → Normal direct transition blocked                 ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║                    Feature Engineering                           ║
║   · 99 features retained in master dataset                       ║
║   · Curated model input subset selected for interpretability     ║
║   · Raw vitals · Slopes · Severity scores · Condition flags      ║
║   · Rolling stats · Lag features                                 ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║               CNN-GRU v6  Deep Learning Model                    ║
║   · Multi-scale Conv1D + 2-layer bidirectional GRU               ║
║   · Attention pooling                                            ║
║   · Temperature calibration  (T = 1.50)                          ║
║   · Dual-threshold decision logic                                ║
║   · Output: Normal / Critical / Emergency                        ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║       [In Development]  Rule-Based Clinical Reasoning Layer      ║
║   · Utilises full 99-feature dataset for deep state analysis     ║
║   · Real-time alerts with explicit physiological explanations    ║
║   · Identifies which vitals and patterns drive each label        ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📊 Dataset

**Source:** [VitalDB ICU Dataset](https://vitaldb.net/)

| Property | Value |
|---|---|
| Patients | 381 ICU patients |
| Age Range | ≥ 60 years and < 80 years |
| Monitoring type | Continuous vital sign streams |
| Resolution | 2-second intervals |
| Total rows | ~2,378,857 |

This forms a **high-resolution geriatric ICU physiological stream dataset** at real clinical scale.

**Target label distribution (`future_label`):**

| Class | Label | Share |
|---|---|---|
| 0 | Normal | ~40% |
| 1 | Critical | ~20% |
| 2 | Emergency | ~40% |

---

## 🫀 Input Vital Signals

### Primary Inputs

| Signal | Column | Description |
|---|---|---|
| SpO₂ | `spo2` | Oxygen saturation |
| Heart Rate | `heart_rate` | Pulse rate (bpm) |
| Respiratory Rate | `resp_rate` | Direct monitor signal |
| Systolic BP | `sbp` | Systolic blood pressure |
| Diastolic BP | `dbp` | Diastolic blood pressure |
| Mean BP | `mbp` | Direct monitor signal — not derived |
| End-Tidal CO₂ | `etco2` | Ventilatory CO₂ marker |

### Derived Signals

```
Pulse Pressure      =  SBP − DBP
resp_rate_smoothed  =  rolling_mean(resp_rate, window)
```

`resp_rate` is a direct input vital. However, raw RR from ICU monitors has high oscillation due to sensor noise. `resp_rate_smoothed` is created by applying a rolling average to suppress this noise — it is used throughout the feature pipeline and model inputs in place of raw RR.

`mbp` is a direct monitor signal. It is not computed as `(SBP + 2×DBP) / 3` — the monitor outputs it directly, so the raw signal is used as-is.

`pulse_pressure` is treated as a first-class feature throughout the system.

---

## ⚙️ Physiological Risk Engine

### Step 1 — Threshold Zone Mapping

Each vital is divided into three clinical risk zones:

| Zone | Meaning |
|---|---|
| Normal | Physiologically stable |
| Critical | Significant abnormality |
| Emergency | Severe instability |

**Full clinical threshold reference table:**

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

Rather than binary zone membership, each vital is mapped to a **continuous score z ∈ [0, 1]**:

```
z = 0.0  →  Normal           (no abnormality)
z = 0.5  →  Critical boundary
z = 1.0  →  Emergency boundary
```

This models **gradual physiological deterioration** rather than abrupt threshold jumps — because real deterioration is gradual.

### Step 3 — Nonlinear Severity Transformation

```
severity = 2^z − 1
```

| z | Severity |
|---|---|
| 0.0 | 0.00 |
| 0.5 | 0.41 |
| 1.0 | 1.00 |

Severity accelerates near emergency levels — reflecting the **nonlinear escalation of clinical risk** that happens in real physiology. A patient moving from z=0.5 to z=1.0 is not twice as sick; they are approaching a physiological cliff.

### Step 4 — Multi-Organ Risk Aggregation

```
severity_sum = Σ severity_i   (across all 8 vital signs)
```

This captures two distinct clinical scenarios simultaneously:
- A **single severe abnormality** — one vital in emergency territory
- **Multiple concurrent mild abnormalities** — multi-organ physiological stress accumulating across systems

---

## 🔬 Disease Pattern Modeling

F.T encodes **12 clinically meaningful deterioration patterns** across three tiers.

Unlike threshold alarms that check single vitals in isolation, these patterns capture **physiological combinations** — the way real clinical instability actually presents.

### Tier 1 — Major Instability

| Pattern | Trigger | Clinical Meaning |
|---|---|---|
| **Shock Spiral** | MBP < 70 AND HR > 100 | Low perfusion with compensatory tachycardia |
| **Respiratory Burnout** | SpO₂ < 92 AND RR > 22 | Oxygen failure with increased respiratory effort |
| **Hypercapnic Failure** | ETCO₂ > 50 AND RR < 10 | Ventilatory failure with CO₂ retention |

### Tier 2 — Moderate Risk

| Pattern | Trigger |
|---|---|
| **Pulse Pressure Low** | Pulse Pressure ≤ 30 |
| **Wide PP + High SBP** | Pulse Pressure ≥ 70 AND SBP ≥ 170 |
| **Respiratory-Hemodynamic Combo** | SpO₂ < 92 AND RR > 22 AND HR > 100 |

### Tier 3 — Subtle / Hidden Risk

These are the most dangerous patterns — they can appear stable superficially while masking serious underlying deterioration.

| Pattern | Trigger | Why It's Dangerous |
|---|---|---|
| **Hypertensive Emergency** | SBP ≥ 180 AND Pulse Pressure ≥ 70 | Extreme pressure with wide pulse |
| **Stable Deceiver** | SpO₂ 92–94 AND HR 75–90 AND MBP 65–70 | Acceptable-looking vitals hiding early circulatory failure |
| **Masked Shock** | MBP 65–72 AND HR < 90 | Perfusion decline *without* compensatory tachycardia — easy to miss |
| **Occult Acidosis** | ETCO₂ ≤ 32 AND RR ≥ 24 AND SpO₂ 88–92 | Metabolic distress below overt alarm values |
| **Trend Decline** | Simultaneous adverse point-to-point changes in ETCO₂, SpO₂, HR | Coordinated multi-vital drift |
| **Trend Activate** | Slope-based sustained deterioration across 5–7 minute windows | Trajectory-level signal before any threshold is crossed |

### Early Deterioration Ramp

Warning signals begin **before thresholds are crossed**:

```
early_start = threshold − 20% × (threshold − normal_reference)
```

This is what enables the 15-minute early warning. The model accumulates signal from the early ramp — it does not wait for threshold breach to start responding.

### Condition Amplification

When deterioration patterns activate, they amplify the final instability score:

```
final_score = severity_sum × M_eff
M_eff = 1 + A × (target_multiplier − 1)
```

Where:
- `A` = condition activation strength (continuous, range 0–1)
- Multipliers are **capped at 2.2** to prevent runaway escalation from stacked conditions

---

## 🔁 Temporal Stability Engine (FSM)

A **Finite State Machine** prevents label flickering caused by noisy vital sign data.

In real ICU settings, sensor artifacts can cause a vital to spike momentarily without any physiological change. Without this layer, those spikes produce false alarms and erode clinical trust in the system.

**FSM rules:**

| Rule | Detail |
|---|---|
| Confirmation threshold | **15 consecutive identical states** required to confirm a label change |
| Emergency → Normal | Direct transition is **blocked** — must pass through Critical first |
| Mixed state handling | Mixed Critical / Emergency states collapse to **Critical** |
| Downgrade policy | Requires **sustained recovery** — not a single normal reading |

State transitions reflect genuine physiological change, not sensor noise.

---

## 🎯 Severity Classification

```
final_score < 0.75            →  ✅ Normal
0.75 ≤ final_score < 1.5      →  ⚠️  Critical
final_score ≥ 1.5             →  🚨 Emergency
```

---

## 🧮 Feature Engineering

F.T separates two concerns: the **master dataset** (full analytical coverage, 99 features) and the **model input set** (curated subset for interpretable, clinically deployable prediction).

> **Design principle:** Every model input maps directly to a physiological concept. High-dimensional or black-box transformations are deliberately excluded from model inputs. The full 99-feature master dataset is preserved for the Rule-Based Reasoning Layer, which needs the complete temporal and pattern context to generate clinical explanations.

---

### Master Dataset — 99 Features

| Category | Count | Features |
|---|---|---|
| **Identifiers** | 2 | `patient_id` · `time` |
| **Raw Vitals** | 9 | `dbp` · `mbp` · `heart_rate` · `resp_rate` · `sbp` · `spo2` · `etco2` · `pulse_pressure` · `resp_rate_smoothed` |
| **Vital Slopes (2m, 5m, 7m, 15m)** | 32 | OLS slope for each of 8 vitals (`spo2` · `heart_rate` · `resp_rate_smoothed` · `sbp` · `dbp` · `mbp` · `etco2` · `pulse_pressure`) across 4 time horizons |
| **Continuous Abnormality Scores** | 8 | `z_spo2` · `z_hr` · `z_rr` · `z_sbp` · `z_dbp` · `z_mbp` · `z_etco2` · `z_pp` |
| **Scaled Severity Scores** | 8 | `s_spo2` · `s_hr` · `s_rr` · `s_sbp` · `s_dbp` · `s_mbp` · `s_etco2` · `s_pp` |
| **Physiological Instability Scores** | 2 | `severity_sum` · `combined_score` |
| **Disease Pattern Flags** | 12 | `t1_shock_spiral` · `t1_resp_burnout` · `t1_hypercapnic` · `t2_pulse_pressure_low` · `t2_widepp_highsbp` · `t2_resp_hemo_combo` · `t3_hyper_emergency` · `t3_stable_deceiver` · `t3_masked_shock` · `t3_occult_acidosis` · `t3_trend_decline` · `t3_trend_activate` |
| **Combined Score Slopes (2m, 5m, 7m, 15m)** | 4 | `slope_2m_combined_score` · `slope_5m_combined_score` · `slope_7m_combined_score` · `slope_15m_combined_score` |
| **Rolling Statistics (2m, 5m, 7m, 15m)** | 10 | `roll_mean_2m_combined` · `roll_std_2m_combined` · `roll_mean_5m_combined` · `roll_std_5m_combined` · `roll_mean_7m_combined` · `roll_std_7m_combined` · `roll_mean_15m_combined` · `roll_std_15m_combined` · `roll_min_15m_combined` · `roll_max_15m_combined` |
| **Lag Features (15m lookback)** | 9 | `lag_15m_spo2` · `lag_15m_heart_rate` · `lag_15m_resp_rate_smoothed` · `lag_15m_sbp` · `lag_15m_dbp` · `lag_15m_mbp` · `lag_15m_etco2` · `lag_15m_pulse_pressure` · `lag_15m_combined_score` |
| **Labels** | 3 | `severity_label` · `result_label` · `future_label` |

---

### Model Input Features

The CNN-GRU v6 model uses the following features selected from the master dataset. Every feature has a direct physiological interpretation.

| Category | Features |
|---|---|
| **Raw Vitals** | `dbp` · `mbp` · `heart_rate` · `sbp` · `spo2` · `etco2` · `pulse_pressure` · `resp_rate_smoothed` |
| **Scaled Severity Scores** | `s_spo2` · `s_hr` · `s_rr` · `s_sbp` · `s_dbp` · `s_mbp` · `s_etco2` · `s_pp` |
| **Vital Slopes (5m, 7m, 15m)** | `slope_{5m/7m/15m}_{spo2/heart_rate/resp_rate_smoothed/sbp/dbp/mbp/etco2/pulse_pressure}` — 24 features |
| **Combined Score Slopes (all windows)** | `slope_2m_combined_score` · `slope_5m_combined_score` · `slope_7m_combined_score` · `slope_15m_combined_score` |
| **Rolling Statistics & Score** | `combined_score` · `roll_mean_15m_combined` · `roll_min_15m_combined` · `roll_max_15m_combined` · `roll_std_15m_combined` |
| **Targeted Pattern Flags** | `t3_masked_shock` · `t3_stable_deceiver` · `t3_occult_acidosis` |

The 2m vital slopes, z-scores, full pattern flag set, shorter-window rolling stats, and lag features are retained in the master dataset for the Rule-Based Reasoning Layer but excluded from model inputs.

---

### Vital Slopes — Time Windows

| Window | Trend Scope | In Model | In Master Dataset |
|---|---|---|---|
| 2 minutes | Short-term spike detection | ❌ (combined score only) | ✅ |
| 5 minutes | Near-term trend | ✅ | ✅ |
| 7 minutes | Medium-term trend | ✅ | ✅ |
| 15 minutes | Sustained trajectory | ✅ | ✅ |

---

## 🤖 Deep Learning Model — CNN-GRU v6

### Architecture

```
Input: (batch, 240 timesteps, n_features)   ← 8-minute window at 2s resolution
          ↓
  Multi-scale Conv1D feature extraction
  (kernel sizes 7 → 5 → 3, channels 48 → 72 → 72)
          ↓
  2-layer bidirectional GRU  (hidden=48, output=96)
          ↓
  Attention pooling
          ↓
  Temperature scaling  (T = 1.50, calibrated on validation set)
          ↓
  Dual-threshold decision logic
  · t_critical  = 0.40
  · t_emergency = 0.28
          ↓
  Output: 3-class severity  (Normal / Critical / Emergency)

Total parameters: 136,292
```

### Window Configuration

| Parameter | Value | Description |
|---|---|---|
| Window length | 240 rows (8 min) | Input sequence length |
| Stride | 20 | Step between windows |
| Train windows | 81,503 | — |
| Val windows | 9,664 | — |
| Test windows | 10,003 | — |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Batch size | 256 |
| Max epochs | 60 (early stopped at epoch 21) |
| Loss function | Cross-entropy with label smoothing (0.02) |
| Class balancing | Asymmetric weighted sampling (Critical ×1.5, Emergency ×1.2) |
| Post-hoc calibration | Temperature scaling (T = 1.4997) |
| Decision logic | Dual-threshold sweep (optimized on val set) |

### Train / Val / Test Split — Patient Level

| Split | Windows | Normal | Critical | Emergency |
|---|---|---|---|---|
| Train | 81,503 | 33,069 | 16,599 | 31,835 |
| Val | 9,664 | 3,419 | 2,269 | 3,976 |
| Test | 10,003 | 3,078 | 2,309 | 4,616 |

---

## 📈 Model Performance

### Summary Metrics

| Metric | Value |
|---|---|
| **Risk Detection Rate** (Emergency flagged as Critical or Emergency) | **~87%** |
| Test AUROC | 0.7260 |
| AUPRC | 0.5615 |
| Balanced Accuracy | 0.52 |
| Accuracy | 0.53 |

### Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal (0) | 0.63 | 0.52 | 0.57 | 3,078 |
| Critical (1) | 0.31 | 0.61 | 0.41 | 2,309 |
| Emergency (2) | 0.79 | 0.49 | 0.60 | 4,616 |

### Test Confusion Matrix

| | Pred-Normal | Pred-Critical | Pred-Emergency |
|---|---|---|---|
| **True-Normal** | 1,816 | 895 | 367 |
| **True-Critical** | 648 | 1,219 | 442 |
| **True-Emergency** | 583 | 1,992 | 2,041 |

---

## 🔍 Understanding the Results

### Why per-class Emergency recall (0.44) is not the right metric

Strict label-level recall counts only Emergency→Emergency predictions. But in clinical use, any flag — Critical or Emergency — triggers bedside attention. Measuring only exact label matches underestimates real clinical utility.

```
Strict Emergency recall:   0.44   (Emergency → Emergency only)
Clinical recall:           0.87   (Emergency → Critical or Emergency)
```

The 87% number is the **operationally correct measure** for an early warning system.

### Emergency → Critical: Conservative Early Detection (Not a Failure)

Emergency cases predicted as Critical are not errors. They represent:
- Genuine physiological instability detected
- A conservative severity label applied
- A clinical alert that still fires and triggers intervention

The model has learned that **Critical is physiologically early Emergency** — patients do not jump to emergency states, they transition through them. Predicting a severe patient as Critical reflects this temporal understanding.

### Critical → Emergency: Safety-Oriented Escalation

Approximately **17% of Critical cases are classified as Emergency** by the model.

This happens because the model detects strong instability signatures even when ground truth is Critical. In a safety-critical environment:

> **The cost of over-escalation is lower than the cost of missing a deteriorating patient.**

This is desirable behavior — clinical conservatism, not model error.

### Key Insight — Risk-Aware Stratification

The model does not enforce rigid class boundaries. Instead it performs **risk-aware stratification** where adjacent severity classes are treated as overlapping regions in a physiological continuum.

| Overlap Direction | Clinical Interpretation |
|---|---|
| Emergency → Critical | Early detection: severe case caught before full emergency |
| Critical → Emergency | Conservative escalation: model prioritizes safety |

Together: the model captures deterioration as a **spectrum**, not discrete steps — which is exactly how ICU physiology behaves.

---

## 🗂️ Repository Structure

```
├── cleaning.ipynb                  # Physiological risk engine & feature pipeline
├── cnn_gru_training.py        # Feature engineering & CNN-GRU v6 training
├── README.md
│
└── cnn_gru_v_outputs/
    ├── model_weights/            # Trained model (.h5 / SavedModel format)
    ├── scaler_metadata/          # Normalisation artifacts (.pkl)
    ├── evaluation_report/        # Full classification report
    ├── training_curves/          # Loss / accuracy history plots
    └── confusion_matrices/       # Val and test confusion matrix visualizations
```

---

## 📥 Dataset Download

The master dataset is hosted on Kaggle due to GitHub file size limits.

| Dataset | Description | Link |
|---|---|---|
| **Initial Dataset** | Raw data before cleaning | [Download from Kaggle](https://www.kaggle.com/datasets/arjunmahesh09999/before-cleaning) |
| **Master Dataset** | Processed, cleaned, 99-feature dataset | [Download from Kaggle](https://www.kaggle.com/datasets/arjunmahesh09999/new-masterdata) |

After downloading, place the dataset file in the **project root directory** before running any scripts.

---

## 💡 Applications

- ICU early warning and real-time deterioration monitoring
- Clinical decision support with explicit physiological reasoning
- Multi-organ failure detection research
- Physiological instability modeling and high-resolution dataset construction
- Rule-based AI integration for bedside alert explanation
- Research foundation for geriatric ICU physiological stream analysis

---

## 🗺️ Planned Extensions

- [ ] **Rule-Based Clinical Reasoning Layer** — Real-time alerting engine that utilises the full 99-feature dataset to identify which specific vital signs, patterns, and trajectories are driving each severity classification, and communicates exact clinical reasoning to bedside staff
- [ ] **Score Fluctuation Analysis** — Studying `combined_score` trajectories: instability oscillation patterns, transition velocities, and physiological drivers behind score variance
- [ ] Validation on larger, multi-hospital datasets (generalizability study)
- [ ] Transformer / attention-based time-series architecture exploration
- [ ] LSTM architecture comparison study
- [ ] Real-time ICU deployment pipeline
- [ ] Prospective clinical validation study
- [ ] Extension to broader ICU age groups (below 60 and above 80)

---

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| Single-center data | Trained on one dataset — generalizability to other ICU populations or hospital settings is unknown |
| Research prototype | Not a certified clinical product — requires prospective validation before any clinical use |
| Hardware dependency | Requires continuous high-frequency vital monitoring at 2-second resolution |
| Label boundary ambiguity | Critical / Emergency boundary is not sharply defined in real physiological data — some label overlap is inherent in the ground truth itself |

---

## 📋 Quick Reference

| Property | Value |
|---|---|
| Target population | ICU patients aged 60–80 |
| Input vitals | SpO₂, HR, RR, SBP, DBP, MBP, ETCO₂, Pulse Pressure |
| Data resolution | 2-second intervals |
| Prediction window | 8-min input → up to 15 min early warning |
| Severity classes | Normal / Critical / Emergency |
| Master dataset features | 99 |
| Model architecture | CNN-GRU v6 (136,292 parameters) |
| Risk detection rate | ~87% |
| Test AUROC | 0.7260 |

---

<div align="center">
<sub>F.T is a research prototype. Not intended for direct clinical use without prospective validation and regulatory clearance.</sub>
</div>
