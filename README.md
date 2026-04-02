# F.T — Flow of Trajectory
### Physiological Instability Detection Engine

> *A time-aware clinical intelligence system for real-time ICU deterioration detection — combining physiological rule modelling, temporal state logic, and deep learning to flag patient instability up to 15 minutes before clinical failure.*

Built on the **VitalDB dataset**: 2,378,857 rows of high-resolution physiological signals from 381 surgical/ICU patients, recorded at 2-second intervals.

---

## What F.T Does — Three Core Purposes

F.T is not a single model. It is a layered clinical intelligence system with three distinct, integrated purposes:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PURPOSE 1 │  Real-Time Severity Classification                      │
│            │  Classifies each patient moment as Normal /             │
│            │  Critical / Emergency using physiological rules         │
│            │  and a temporal state engine                            │
├─────────────────────────────────────────────────────────────────────┤
│  PURPOSE 2 │  Rule-Based Explainability                              │
│            │  Explains why a patient is classified the way they are  │
│            │  — in plain clinical language, tracing every score      │
│            │  back to specific vitals, thresholds, and patterns      │
├─────────────────────────────────────────────────────────────────────┤
│  PURPOSE 3 │  15-Minute Ahead Deterioration Prediction               │
│            │  A deep learning model (CNN-GRU v7) trained to predict  │
│            │  severity class 15 minutes into the future, enabling    │
│            │  pre-emptive clinical intervention                       │
└─────────────────────────────────────────────────────────────────────┘
```

Each layer is independently useful. Together, they form a complete early warning system.

---

## Performance at a Glance

```
Emergency Detection Rate        ~94%
Binary AUROC (Normal vs At-Risk)  0.7987
Test AUROC                        0.7234
Model Parameters                  275,396
Prediction Horizon                ~15 minutes ahead
Input Resolution                  2-second vital streams
```

### What Does 94% Actually Mean?

This is not accuracy. It is not recall. It is the answer to the only question that matters in an early warning system:

> **"When a patient is deteriorating — does the system flag it?"**

```
Emergency → Emergency    1,757  ✅  Direct detection
Emergency → Critical     1,646  ✅  Conservative early detection (still triggers intervention)
Emergency → Missed          71  ❌  Missed
──────────────────────────────────────────────────────
Clinical detection rate   3,403 / 3,474  ≈  94%
```

When the model labels a deteriorating patient as **Critical** instead of **Emergency**, that is not an error — it is **conservative early detection**. The alert still fires. The clinician is still called. The patient still receives attention. Both labels trigger intervention; only one is penalised by conventional metrics.

---

## Why Not NEWS2?

The National Early Warning Score (NEWS2) is the most widely adopted clinical deterioration tool in the world. So why build F.T?

Because NEWS2 detects deterioration **after it becomes obvious**. F.T detects it **as it develops**.

| Capability | NEWS2 | F.T |
|---|---|---|
| Approach | Static snapshot scoring | Continuous physiological trajectory |
| Signal source | Current threshold breaches | Evolving multi-vital patterns |
| Pattern detection | Single vital, isolated | Multi-vital combinations across organ systems |
| Temporal consistency | None — single point in time | FSM-stabilised 15-window confirmation |
| Early warning | At threshold breach | Before threshold breach (ramp detection) |
| Hidden instability | Not detected | Tier 3 patterns: Masked Shock, Stable Deceiver, Occult Acidosis |

### The NEWS2 Comparison

F.T was benchmarked against a modified NEWS2 implementation adapted for the intraoperative data constraints of VitalDB. Temperature and consciousness were excluded (not consistently available or meaningful under anaesthesia); respiratory rate, SpO₂, heart rate, SBP, and FiO₂ proxy were used.

| Metric | Agreement |
|---|---|
| Severity Match | 61.39% (33,612 / 54,750 rows) |
| Result Match | 60.86% (33,323 rows) |
| Exact Match (Both) | 54.10% (29,618 rows) |

The ~54% exact match is **expected and informative**. It reflects the fundamental difference in philosophy between the two systems. NEWS2 asks *"Is this patient currently outside normal bounds?"* F.T asks *"Is this patient's physiological trajectory heading toward failure?"*

### What NEWS2 Cannot See

F.T's Tier 3 patterns are specifically designed to detect instability that NEWS2 structurally cannot:

- **Stable Deceiver** — SpO₂ 92–94%, HR 75–90, MBP 65–70: vitals that look acceptable while masking early circulatory failure
- **Masked Shock** — MBP 65–72, HR < 90: perfusion decline without the compensatory tachycardia that would normally raise an alarm
- **Occult Acidosis** — ETCO₂ ≤ 32, RR ≥ 24, SpO₂ 88–92: metabolic distress where each individual vital sits just below any single-vital alert threshold

These are the presentations where patients deteriorate undetected. F.T is designed to catch them.

> **F.T does not replace NEWS2. It extends it** — adding early and hidden instability detection, multi-parameter physiological interaction modelling, and time-consistent severity estimation that a snapshot score cannot provide.

---

## Full System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║          Raw ICU Vital Streams  (2-second resolution)            ║
║  SpO₂ · HR · RR · SBP · DBP · MBP · ETCO₂ · Pulse Pressure     ║
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
║   · Label confirmation across 15 consecutive readings            ║
║   · Prevents sensor-noise-driven label flickering                ║
║   · Emergency → Normal direct transition blocked                 ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
╔══════════════════════╗     ╔══════════════════════════════════╗
║  PURPOSE 1 & 2       ║     ║  PURPOSE 3                       ║
║  Real-Time Severity  ║     ║  Feature Engineering (99 feats)  ║
║  + Rule-Based XAI    ║     ║  → CNN-GRU v7 Deep Learning      ║
║                      ║     ║  → 15-Min Ahead Prediction       ║
╚══════════════════════╝     ╚══════════════════════════════════╝
```

---

## Purpose 1 — Real-Time Severity Classification

### Physiological Risk Engine

**Step 1 — Threshold Zone Mapping**

Each vital is divided into three clinical risk zones:

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

**Step 2 — Continuous Abnormality Encoding**

Rather than binary zone membership (normal / not normal), each vital is mapped to a continuous score `z ∈ [0, 1]`. This models gradual physiological deterioration — a patient sliding toward the emergency threshold is represented as increasingly abnormal, not abruptly classified when they cross a line.

```
z = 0.0  →  Normal (no abnormality)
z = 0.5  →  Critical boundary
z = 1.0  →  Emergency boundary
```

**Step 3 — Nonlinear Severity Transformation**

```
severity = 2^z − 1
```

| z | Severity |
|---|---|
| 0.0 | 0.00 |
| 0.5 | 0.41 |
| 1.0 | 1.00 |

Severity accelerates near emergency levels — reflecting the clinically real phenomenon of nonlinear risk escalation as patients approach physiological failure.

**Step 4 — Multi-Organ Risk Aggregation**

```
severity_sum = Σ severity_i   (across all 8 vital signs)
```

This captures two distinct clinical scenarios simultaneously: a single severely deranged vital, and multiple concurrent mild abnormalities spanning organ systems.

### Disease Pattern Modelling — 12 Clinical Patterns Across 3 Tiers

F.T encodes physiological combinations — the way real clinical instability presents — rather than single-vital threshold alarms.

**Tier 1 — Major Instability**

| Pattern | Trigger | Clinical Meaning |
|---|---|---|
| Shock Spiral | MBP < 70 AND HR > 100 | Low perfusion with compensatory tachycardia |
| Respiratory Burnout | SpO₂ < 92 AND RR > 22 | Oxygen failure with increased respiratory effort |
| Hypercapnic Failure | ETCO₂ > 50 AND RR < 10 | Ventilatory failure with CO₂ retention |

**Tier 2 — Moderate Risk**

| Pattern | Trigger |
|---|---|
| Pulse Pressure Low | Pulse Pressure ≤ 30 |
| Wide PP + High SBP | Pulse Pressure ≥ 70 AND SBP ≥ 170 |
| Respiratory-Hemodynamic Combo | SpO₂ < 92 AND RR > 22 AND HR > 100 |

**Tier 3 — Subtle / Hidden Risk**

These are the most clinically dangerous patterns. They appear stable superficially while masking serious underlying deterioration. These are the presentations that NEWS2 misses.

| Pattern | Trigger | Why It's Dangerous |
|---|---|---|
| Hypertensive Emergency | SBP ≥ 180 AND PP ≥ 70 | Extreme pressure with wide pulse |
| Stable Deceiver | SpO₂ 92–94 AND HR 75–90 AND MBP 65–70 | Acceptable-looking vitals hiding early circulatory failure |
| Masked Shock | MBP 65–72 AND HR < 90 | Perfusion decline without compensatory tachycardia — easy to miss |
| Occult Acidosis | ETCO₂ ≤ 32 AND RR ≥ 24 AND SpO₂ 88–92 | Metabolic distress below overt alarm values |
| Trend Decline | Simultaneous adverse point-to-point changes in ETCO₂, SpO₂, HR | Coordinated multi-vital drift |
| Trend Activate | Slope-based sustained deterioration across 5–7 min windows | Trajectory-level signal before any threshold is crossed |

### Severity Classification Thresholds

```
final_score < 0.75            →  ✅ Normal
0.75 ≤ final_score < 1.5      →  ⚠️  Critical
final_score ≥ 1.5             →  🚨 Emergency
```

### Temporal Stability Engine (FSM)

A Finite State Machine prevents label flickering from noisy vital sign data.

| Rule | Detail |
|---|---|
| Confirmation threshold | 15 consecutive identical states required to confirm a label change |
| Emergency → Normal | Direct transition blocked — must pass through Critical first |
| Mixed state handling | Mixed Critical / Emergency states collapse to Critical |
| Downgrade policy | Requires sustained recovery — not a single normal reading |

---

## Purpose 2 — Rule-Based Explainability Layer

The ML model produces a risk score. The rule-based layer explains why.

Modern clinical AI models often produce risk scores without explaining their reasoning — a fundamental barrier to adoption in medical settings. The rule-based layer solves this by adding a fully transparent, deterministic explainability pipeline on top of the severity classification.

For every patient time point, it generates a structured clinical report that:
- Confirms or challenges the model's severity label using deterministic rules
- Identifies which specific vitals crossed which thresholds
- Detects multi-vital clinical pattern combinations
- Warns about monotonic deterioration trends and physiologically impossible sensor values
- Identifies early recovery signals across multiple time windows

Every output is traceable to a specific vital, a specific threshold, and a specific clinical rationale. There is no black-box inference.

### 5-Stage Explainability Pipeline

```
Patient Snapshot (patient_id + time)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — System Label Display                          │
│  Shows severity_label vs result_label side by side       │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — FSM Mismatch Explanation                      │
│  Explains why the confirmed label may differ from raw    │
│  (e.g., a transient spike that did not persist)          │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — Vital Sign Analysis                           │
│  A. Emergency-threshold triggers (per-vital breakdown)   │
│  B. Active multi-vital clinical pattern flags            │
│  C. All out-of-range vitals (comprehensive list)         │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4 — Trend & Noise Warnings                        │
│  · Monotonic slope detection across 15m → 7m → 5m → 2m  │
│  · Physiologically impossible value detection            │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 5 — Recovery Detection                            │
│  · Vitals trending consistently back toward normal       │
│  · Requires 15-minute lag snapshot for comparison        │
└──────────────────────────────────────────────────────────┘
        │
        ▼
  Structured Clinical Report
```

**Why four nested time windows for trend analysis?**
A single slope (15m → now) can be misleading if the trend reversed halfway. Checking across four nested windows (15m → 7m → 5m → 2m) ensures that only genuinely continuous or accelerating trends are flagged as deterioration warnings.

**Why a FSM for label confirmation rather than just the raw score?**
Transient spikes — a single high reading from patient movement, probe displacement, or momentary physiological variation — must not trigger emergency alerts. The FSM requires 15 consecutive readings in the same severity state before a label change is confirmed. This separates genuine sustained deterioration from noise.

### Example Report Output

```
═════════════════════════════════════════════════════════════════
     RULE-BASED EXPLAINABLE AI LAYER — PATIENT VITAL ANALYSIS
═════════════════════════════════════════════════════════════════
  Patient ID      : 64
  Time Point      : 3000s  (50 min into monitoring)
  Combined Score  : 4.7562
─────────────────────────────────────────────────────────────────
  STAGE 1 — SYSTEM LABELS
  Patient Current Condition  : Emergency   (severity_label)
  System Confirmed Condition : Emergency   (result_label)
─────────────────────────────────────────────────────────────────
  STAGE 2 — CONDITION MISMATCH EXPLANATION

  🚨 Both the raw severity and the system-confirmed label agree:
     the patient is in a confirmed EMERGENCY state.
     Deterioration has been sustained across the 15-reading
     evaluation window. The FSM has locked in this label.
     Clinical attention is required.
═════════════════════════════════════════════════════════════════
  STAGE 3 — VITAL SIGN ANALYSIS

  STEP A — EMERGENCY-THRESHOLD VITAL TRIGGERS

  🔴 Respiratory Rate
       Current Value  : 40.0 breaths/min
       Status         : Emergency — above expected range of 20 /min
       Clinical Note  : Patient is breathing faster than normal.
                        Body may be compensating for low oxygen or acidosis.

  🔴 Systolic Blood Pressure
       Current Value  : 89.0 mmHg
       Status         : Emergency — below expected range of 110 mmHg
       Clinical Note  : Hypotension — heart may not be pumping
                        sufficient blood to vital organs.

  🔴 Mean Arterial Pressure
       Current Value  : 62.0 mmHg
       Status         : Critical — below expected range of 70 mmHg
       Clinical Note  : MAP below 70 mmHg is associated with
                        organ ischemia and shock.

  🔴 Pulse Pressure
       Current Value  : 43.0 mmHg
       Status         : Borderline-Low
       Clinical Note  : Critical warning sign of tamponade,
                        severe hypovolemia, or cardiogenic shock.
─────────────────────────────────────────────────────────────────
  STAGE 4 — TREND & NOISE WARNINGS
  ✅ No continuous monotonic trend detected across lag vitals.
  ✅ No impossible vital values or signal noise detected.
─────────────────────────────────────────────────────────────────
  STAGE 5 — RECOVERY ANALYSIS
  ℹ️  No recovery signs detected at this time.
═════════════════════════════════════════════════════════════════
         FINAL REPORT — Patient 64 @ 3000s
═════════════════════════════════════════════════════════════════
  Raw Severity       : EMERGENCY
  FSM Confirmed      : EMERGENCY
  15-Min Forecast    : Emergency (probability 1.0)

  🚨  CONFIRMED EMERGENCY — Raw severity and FSM both agree.
      Sustained deterioration confirmed. Medical action required.
═════════════════════════════════════════════════════════════════
```

---

## Purpose 3 — 15-Minute Ahead Deterioration Prediction

### How Early Warning Is Achieved

Standard threshold-based systems (including NEWS2) trigger at the moment a vital crosses a boundary. F.T begins accumulating signal **before** that boundary is reached, through a mechanism called the **early deterioration ramp**.

```
early_start = threshold − 20% × (threshold − normal_reference)
```

This means the system starts encoding abnormality before the vital reaches any formal threshold. The slope features, rolling statistics, and combined score trajectories fed to the CNN-GRU model carry this sub-threshold signal forward — allowing the model to predict deterioration based on trajectory rather than position.

### CNN-GRU v7 Architecture

```
Input: (batch, 80 timesteps × 44 features)
          ↓
  Multi-scale Conv1D
  (kernel sizes 7 → 5 → 3, channels 48 → 72 → 72)
  + Residual connections
          ↓
  2-layer Bidirectional GRU  (hidden=48, output=96)
          ↓
  Attention Pooling
          ↓
  Temperature Scaling  (T = 1.49)
          ↓
  Output: Normal / Critical / Emergency  (15 min ahead)
```

Total parameters: **275,396**

**Why CNN + GRU?**
The multi-scale Conv1D layers extract local physiological patterns at different time scales simultaneously (short transients vs sustained trends). The bidirectional GRU reads these patterns sequentially, capturing how the physiological state evolves over the 80-step (roughly 2.7-minute) input window. Attention pooling allows the model to weight which timesteps matter most for the prediction. The residual connections allow gradient flow through the convolutional stack without degradation.

**Why temperature scaling?**
A calibration temperature of T = 1.49 was applied post-hoc. Without calibration, the model's softmax probabilities tend to be overconfident — inflated toward extreme values. Temperature scaling produces well-calibrated probability estimates, which is important in a clinical setting where the degree of certainty matters, not just the predicted class.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Batch size | 256 |
| Max epochs | 60 (early stopped at epoch 29) |
| Loss function | Focal Loss |
| Regularization | SWA + EMA smoothing + Jitter augmentation |
| Post-hoc calibration | Temperature scaling (T = 1.49) |

**Why Focal Loss?**
The dataset has a significant class imbalance (Normal ~40%, Critical ~20%, Emergency ~40%). Focal Loss down-weights well-classified examples during training, forcing the model to focus on hard cases — which in a clinical context tend to be the Critical/Emergency boundary cases that matter most.

### Dataset Split — Patient-Level

Splits are patient-level, not row-level. This prevents data leakage (a patient appearing in both train and test splits would inflate performance metrics).

| Split | Windows | Normal | Critical | Emergency |
|---|---|---|---|---|
| Train | 57,300 | 23,028 | 11,660 | 22,612 |
| Val | 6,630 | 2,323 | 1,583 | 2,724 |
| Test | 6,876 | 2,109 | 1,547 | 3,220 |

### Full Performance Metrics

**Summary**

| Metric | Value |
|---|---|
| Emergency Detection Rate (flagged as Critical or Emergency) | ~94% |
| Binary AUROC (Normal vs Critical+Emergency) | 0.7987 |
| Test AUROC | 0.7234 |
| AUPRC | 0.5654 |
| Balanced Accuracy | 0.53 |

**Per-Class Breakdown**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal (0) | 0.69 | 0.42 | 0.52 | 2,109 |
| Critical (1) | 0.30 | 0.64 | 0.41 | 1,547 |
| Emergency (2) | 0.75 | 0.55 | 0.63 | 3,220 |

**Confusion Matrix**

| | Pred Normal | Pred Critical | Pred Emergency |
|---|---|---|---|
| True Normal | 890 | 978 | 241 |
| True Critical | 215 | 985 | 347 |
| True Emergency | 71 | 1,646 | 1,757 |

**Reading the confusion matrix correctly:**
The Critical class has low precision (0.30) because the boundary between Critical and Emergency is physiologically ambiguous — ground truth labels themselves carry uncertainty at this boundary. The system is intentionally tuned toward recall (catching deteriorating patients) at the cost of precision. In a safety-critical clinical context, a false alarm is far less costly than a missed deterioration.

---

## Feature Engineering

F.T separates two concerns: the **master dataset** (full analytical coverage, 99 features) and the **model input set** (curated interpretable subset).

**Design principle:** Every model input maps directly to a physiological concept. High-dimensional transformations, PCA, latent embeddings, and black-box statistical constructs are deliberately excluded from model inputs. The system can explain every feature it acts on.

### Master Dataset — 99 Features

| Category | Count | Features |
|---|---|---|
| Identifiers | 2 | patient_id · time |
| Raw Vitals | 9 | dbp · mbp · heart_rate · resp_rate · sbp · spo2 · etco2 · pulse_pressure · resp_rate_smoothed |
| Vital Slopes (2m, 5m, 7m, 15m) | 32 | OLS slope for each of 8 vitals across 4 time horizons |
| Continuous Abnormality Scores | 8 | z_spo2 · z_hr · z_rr · z_sbp · z_dbp · z_mbp · z_etco2 · z_pp |
| Scaled Severity Scores | 8 | s_spo2 · s_hr · s_rr · s_sbp · s_dbp · s_mbp · s_etco2 · s_pp |
| Physiological Instability Scores | 2 | severity_sum · combined_score |
| Disease Pattern Flags | 12 | t1_shock_spiral · t1_resp_burnout · t1_hypercapnic · t2_pulse_pressure_low · t2_widepp_highsbp · t2_resp_hemo_combo · t3_hyper_emergency · t3_stable_deceiver · t3_masked_shock · t3_occult_acidosis · t3_trend_decline · t3_trend_activate |
| Combined Score Slopes (2m, 5m, 7m, 15m) | 4 | slope_{2m/5m/7m/15m}_combined_score |
| Rolling Statistics | 10 | roll_mean/std across multiple windows + roll_min/max_15m_combined |
| Lag Features (15m lookback) | 9 | lag_15m for all 8 vitals + lag_15m_combined_score |
| Labels | 3 | severity_label · result_label · future_label |

The 2m vital slopes, z-scores, full pattern flag set, shorter-window rolling stats, and lag features are retained in the master dataset for the Rule-Based Reasoning Layer but excluded from the CNN-GRU model inputs. The model uses a curated subset; the explainability layer uses everything.

---

## Dataset

| Property | Value |
|---|---|
| Source | VitalDB — high-resolution perioperative physiological monitoring |
| Patients | 381 ICU patients |
| Age range | 60–80 years |
| Monitoring type | Continuous vital sign streams |
| Resolution | 2-second intervals |
| Total rows | ~2,378,857 |

**Input Vital Signals**

| Signal | Column | Notes |
|---|---|---|
| SpO₂ | spo2 | Oxygen saturation |
| Heart Rate | heart_rate | Pulse rate (bpm) |
| Respiratory Rate | resp_rate | Direct monitor signal; smoothed to suppress noise |
| Systolic BP | sbp | Systolic blood pressure |
| Diastolic BP | dbp | Diastolic blood pressure |
| Mean BP | mbp | Direct monitor output — not derived as (SBP + 2×DBP)/3 |
| End-Tidal CO₂ | etco2 | Ventilatory CO₂ marker |
| Pulse Pressure | pulse_pressure | Derived as SBP − DBP; treated as a first-class feature |

**Target label distribution (`future_label`):**

| Class | Label | Share |
|---|---|---|
| 0 | Normal | ~40% |
| 1 | Critical | ~20% |
| 2 | Emergency | ~40% |

---

## Repository Structure

```
├── cleaning.ipynb          # Physiological risk engine & feature pipeline
├── cnn_gru_2.py            # Feature engineering & CNN-GRU training
├── RULE_BASED_AI.ipynb     # Rule-Based Clinical Reasoning Layer
├── README.md
├── news-2/                 # NEWS2 labelling and comparison analysis
└── tier_combination/       # Tier pattern explanation and documentation
```

---

## Dataset Download

The master dataset is hosted on Kaggle due to GitHub file size limits. Place downloaded files in the project root before running any scripts.

| Dataset | Description | Link |
|---|---|---|
| Initial Dataset | Raw data before cleaning | [Download from Kaggle](https://www.kaggle.com) |
| Master Dataset | Processed, cleaned, 99-feature dataset | [Download from Kaggle](https://www.kaggle.com) |

---

## Running the Rule-Based Layer

```python
import pandas as pd

# Configuration
TARGET_PATIENT_ID = 64
TARGET_TIME       = 3600   # seconds (3600 = 60 min into monitoring)

# Load data
df = pd.read_csv("MASTERDATA.csv")

# Extract snapshots
current_stat = df[
    (df['patient_id'] == TARGET_PATIENT_ID) & 
    (df['time'] == TARGET_TIME)
].reset_index(drop=True)

lag_stat = df[
    (df['patient_id'] == TARGET_PATIENT_ID) & 
    (df['time'] == TARGET_TIME - 900)
].reset_index(drop=True)

past_stat = None if lag_stat.empty else lag_stat

# Run all stages
stage_1_2(current_stat)
stage_3(current_stat)
stage_4(current_stat)
stage_5(current_stat, past_stat)
final_report(current_stat, past_stat)
```

**Dependencies:** `numpy`, `pandas` — no ML frameworks required. The rule-based layer is fully deterministic.

---

## Limitations

| Limitation | Detail |
|---|---|
| Low Critical precision | Reflects inherent label boundary ambiguity — Critical and Emergency overlap physiologically. The system is intentionally tuned toward recall over precision. |
| Single-center data | Trained on one dataset. Generalizability to other ICU populations or hospital settings is unknown. |
| No clinical validation | Research prototype. Not prospectively validated in a clinical environment. Not a certified medical device. |
| Intraoperative context | VitalDB captures surgical/perioperative monitoring; dynamics may differ from general ICU populations. |
| Hardware dependency | Requires continuous high-frequency vital monitoring at 2-second resolution — not available in all clinical settings. |
| NEWS2 comparison scope | Modified NEWS2 excluded temperature and consciousness due to data constraints; full NEWS2 comparison is pending. |

---

## Planned Extensions

- **XAI Integration** — SHAP / attention visualisation to complement rule-based explanations with feature attribution from the deep learning model
- **Score Fluctuation Analysis** — Studying combined_score trajectories: instability oscillation patterns, transition velocities, and physiological drivers behind score variance
- **Transformer architecture** — Attention-based architecture exploration for extended datasets
- **LSTM comparison study** — Benchmarking against the CNN-GRU architecture
- **Real-time ICU deployment pipeline**
- **Prospective clinical validation study**
- **Multi-hospital validation** — Generalizability across different ICU populations
- **Extended age range** — Validation below 60 and above 80 years

---

## Quick Reference

| Property | Value |
|---|---|
| Target population | ICU patients aged 60–80 |
| Input vitals | SpO₂, HR, RR, SBP, DBP, MBP, ETCO₂, Pulse Pressure |
| Data resolution | 2-second intervals |
| Prediction window | ~2.7-min input → up to 15 min early warning |
| Severity classes | Normal / Critical / Emergency |
| Master dataset features | 99 |
| Model architecture | CNN-GRU v7 (275,396 parameters) |
| Emergency detection rate | ~94% |
| Binary AUROC (Normal vs at-risk) | 0.7987 |
| Test AUROC | 0.7234 |

---

> **F.T is a research prototype. It is not intended for direct clinical use without prospective validation and regulatory clearance.**
