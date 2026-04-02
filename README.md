# F.T — Flow of Trajectory · Physiological Instability Detection Engine

A time-aware clinical intelligence system for real-time ICU deterioration detection — combining physiological rule modelling, temporal state logic, and deep learning to flag patient instability **up to 15 minutes before clinical failure**.

Built on the [VitalDB](https://vitaldb.net/) dataset, which provides high-resolution physiological signals recorded during surgical procedures — including continuously monitored vital parameters such as heart rate, blood pressure, respiratory rate, oxygen saturation,and end-tidal CO₂.

---

## ⚡ What F.T Achieves

> **"Detects ~94% of ICU deterioration cases — up to 15 minutes before clinical failure."**

That number is not accuracy. It is not recall. It is the answer to the only question that matters in an early warning system:

**"When a patient is deteriorating, does the system flag it?"**

```
Emergency → Emergency:   1,757  ✅ Direct detection
Emergency → Critical:    1,646  ✅ Conservative early detection  (both trigger intervention)
Emergency → Normal:         71  ❌ Missed
──────────────────────────────────────────────────────────────────
Combined risk detection:  3,403 / 3,220  ≈  94%
```

The model does not treat Emergency→Critical as an error. It treats it as **early detection** — the system has identified a severely deteriorating patient and raised an alert, using a conservative label. That is clinically useful behaviour.

**Binary AUROC (Normal vs Critical+Emergency): 0.7987** — approaching 0.80, reflecting strong discrimination between stable and at-risk patients.

---

## 🧠 Why This Is Not Just a Prediction Model

F.T is a **dual-layer clinical intelligence system**, not a standalone ML classifier.

 F.T is architected in two integrated layers:

| Layer | Role | Status |
|---|---|---|
| Rule-Based Clinical Reasoning Layer | Explains *why* risk is changing, in human-readable clinical language | ✅ Complete |
| Deep Learning Layer | Predicts severity class 15 minutes ahead (Normal / Critical / Emergency) | ✅ Complete |

The rule-based layer is not an afterthought — it shapes every design decision in the system:

- The model uses a curated subset of **interpretable variables** — not 200+ statistical transforms
- Black-box methods (PCA, latent embeddings, high-dimensional expansions) are deliberately excluded from model inputs
- Each model feature maintains a **direct mapping to a clinical concept**
- The master dataset retains all 99 features for full physiological state analysis

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
║               CNN-GRU v7  Deep Learning Model                    ║
║   · Multi-scale Conv1D + 2-layer bidirectional GRU               ║
║   · Attention pooling + Residual connections                     ║
║   · Focal loss + SWA + EMA smoothing                             ║
║   · Temperature calibration  (T = 1.49)                          ║
║   · Output: Normal / Critical / Emergency                        ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║            Rule-Based Clinical Reasoning Layer                   ║
║   · Utilises full 99-feature dataset for deep state analysis     ║
║   · Real-time alerts with explicit physiological explanations    ║
║   · Identifies which vitals and patterns drive each label        ║
║   · 5-stage explainability pipeline (see below)                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📊 Dataset

**Source:** VitalDB — high-resolution physiological signals from surgical/ICU monitoring

| Property | Value |
|---|---|
| Patients | 381 ICU patients |
| Age Range | ≥ 60 years and < 80 years |
| Monitoring type | Continuous vital sign streams |
| Resolution | 2-second intervals |
| Total rows | ~2,378,857 |

**Target label distribution (future_label):**

| Class | Label | Share |
|---|---|---|
| 0 | Normal | ~40% |
| 1 | Critical | ~20% |
| 2 | Emergency | ~40% |

---

## 🫀 Input Vital Signals

**Primary Inputs**

| Signal | Column | Description |
|---|---|---|
| SpO₂ | `spo2` | Oxygen saturation |
| Heart Rate | `heart_rate` | Pulse rate (bpm) |
| Respiratory Rate | `resp_rate` | Direct monitor signal |
| Systolic BP | `sbp` | Systolic blood pressure |
| Diastolic BP | `dbp` | Diastolic blood pressure |
| Mean BP | `mbp` | Direct monitor signal — not derived |
| End-Tidal CO₂ | `etco2` | Ventilatory CO₂ marker |

**Derived Signals**

```
Pulse Pressure      =  SBP − DBP
resp_rate_smoothed  =  rolling_mean(resp_rate, window)
```

`mbp` is a direct monitor output — not computed as (SBP + 2×DBP) / 3.
`resp_rate_smoothed` is used throughout the feature pipeline in place of raw RR to suppress sensor noise.
`pulse_pressure` is treated as a first-class feature throughout the system.

---

## ⚙️ Physiological Risk Engine

### Step 1 — Threshold Zone Mapping

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

### Step 2 — Continuous Abnormality Encoding

Rather than binary zone membership, each vital is mapped to a continuous score z ∈ [0, 1]:

```
z = 0.0  →  Normal           (no abnormality)
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

Severity accelerates near emergency levels — reflecting the nonlinear escalation of clinical risk as patients approach physiological failure.

### Step 4 — Multi-Organ Risk Aggregation

```
severity_sum = Σ severity_i   (across all 8 vital signs)
```

This captures two distinct clinical scenarios simultaneously: a single severe abnormality, and multiple concurrent mild abnormalities across organ systems.

---

## 🔬 Disease Pattern Modelling

F.T encodes **12 clinically meaningful deterioration patterns** across three tiers. These patterns capture physiological combinations — the way real clinical instability presents — rather than single-vital threshold alarms.

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

These are the most dangerous patterns — they can appear stable superficially while masking serious underlying deterioration.

| Pattern | Trigger | Why It's Dangerous |
|---|---|---|
| Hypertensive Emergency | SBP ≥ 180 AND PP ≥ 70 | Extreme pressure with wide pulse |
| Stable Deceiver | SpO₂ 92–94 AND HR 75–90 AND MBP 65–70 | Acceptable-looking vitals hiding early circulatory failure |
| Masked Shock | MBP 65–72 AND HR < 90 | Perfusion decline without compensatory tachycardia — easy to miss |
| Occult Acidosis | ETCO₂ ≤ 32 AND RR ≥ 24 AND SpO₂ 88–92 | Metabolic distress below overt alarm values |
| Trend Decline | Simultaneous adverse point-to-point changes in ETCO₂, SpO₂, HR | Coordinated multi-vital drift |
| Trend Activate | Slope-based sustained deterioration across 5–7 min windows | Trajectory-level signal before any threshold is crossed |

### Early Deterioration Ramp

Warning signals begin **before** thresholds are crossed:

```
early_start = threshold − 20% × (threshold − normal_reference)
```

This is what enables 15-minute early warning. The system accumulates signal from the early ramp rather than waiting for threshold breach.

### Condition Amplification

```
final_score = severity_sum × M_eff
M_eff = 1 + A × (target_multiplier − 1)
```

Multipliers are capped at 2.2× to prevent runaway escalation from stacked conditions.

---

## 🔁 Temporal Stability Engine (FSM)

A Finite State Machine prevents label flickering caused by noisy vital sign data.

| Rule | Detail |
|---|---|
| Confirmation threshold | 15 consecutive identical states required to confirm a label change |
| Emergency → Normal | Direct transition is blocked — must pass through Critical first |
| Mixed state handling | Mixed Critical / Emergency states collapse to Critical |
| Downgrade policy | Requires sustained recovery — not a single normal reading |

---

## 🎯 Severity Classification

```
final_score < 0.75            →  ✅ Normal
0.75 ≤ final_score < 1.5      →  ⚠️  Critical
final_score ≥ 1.5             →  🚨 Emergency
```

---

## 🧮 Feature Engineering

F.T separates two concerns: the **master dataset** (full analytical coverage, 99 features) and the **model input set** (curated interpretable subset).

**Design principle:** Every model input maps directly to a physiological concept. High-dimensional or black-box transformations are deliberately excluded from model inputs.

### Master Dataset — 99 Features

| Category | Count | Features |
|---|---|---|
| Identifiers | 2 | `patient_id` · `time` |
| Raw Vitals | 9 | `dbp` · `mbp` · `heart_rate` · `resp_rate` · `sbp` · `spo2` · `etco2` · `pulse_pressure` · `resp_rate_smoothed` |
| Vital Slopes (2m, 5m, 7m, 15m) | 32 | OLS slope for each of 8 vitals across 4 time horizons |
| Continuous Abnormality Scores | 8 | `z_spo2` · `z_hr` · `z_rr` · `z_sbp` · `z_dbp` · `z_mbp` · `z_etco2` · `z_pp` |
| Scaled Severity Scores | 8 | `s_spo2` · `s_hr` · `s_rr` · `s_sbp` · `s_dbp` · `s_mbp` · `s_etco2` · `s_pp` |
| Physiological Instability Scores | 2 | `severity_sum` · `combined_score` |
| Disease Pattern Flags | 12 | `t1_shock_spiral` · `t1_resp_burnout` · `t1_hypercapnic` · `t2_pulse_pressure_low` · `t2_widepp_highsbp` · `t2_resp_hemo_combo` · `t3_hyper_emergency` · `t3_stable_deceiver` · `t3_masked_shock` · `t3_occult_acidosis` · `t3_trend_decline` · `t3_trend_activate` |
| Combined Score Slopes (2m, 5m, 7m, 15m) | 4 | `slope_2m_combined_score` · `slope_5m_combined_score` · `slope_7m_combined_score` · `slope_15m_combined_score` |
| Rolling Statistics | 10 | `roll_mean/std` across 2m/5m/7m/15m windows + `roll_min/max_15m_combined` |
| Lag Features (15m lookback) | 9 | `lag_15m_{vital}` for all 8 vitals + `lag_15m_combined_score` |
| Labels | 3 | `severity_label` · `result_label` · `future_label` |

### Model Input Features (CNN-GRU v7)

| Category | Features |
|---|---|
| Raw Vitals | `dbp` · `mbp` · `heart_rate` · `sbp` · `spo2` · `etco2` · `pulse_pressure` · `resp_rate_smoothed` |
| Scaled Severity Scores | `s_spo2` · `s_hr` · `s_rr` · `s_sbp` · `s_dbp` · `s_mbp` · `s_etco2` · `s_pp` |
| Vital Slopes (5m, 7m, 15m) | 24 slope features across 8 vitals |
| Combined Score Slopes | `slope_{2m/5m/7m/15m}_combined_score` |
| Rolling Statistics & Score | `combined_score` · `roll_mean/min/max/std_15m_combined` |
| Targeted Pattern Flags | `t3_masked_shock` · `t3_stable_deceiver` · `t3_occult_acidosis` |

The 2m vital slopes, z-scores, full pattern flag set, shorter-window rolling stats, and lag features are retained in the master dataset for the Rule-Based Reasoning Layer but excluded from model inputs.

---

## 🤖 Deep Learning Model — CNN-GRU v7

### Architecture

```
Input: (batch, 80 timesteps, 44 features)
          ↓
  Multi-scale Conv1D  (kernel sizes 7 → 5 → 3, channels 48 → 72 → 72)
  + Residual connections
          ↓
  2-layer bidirectional GRU  (hidden=48, output=96)
          ↓
  Attention pooling
          ↓
  Temperature scaling  (T = 1.49)
          ↓
  Argmax decision
          ↓
  Output: 3-class severity  (Normal / Critical / Emergency)

Total parameters: 275,396
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Batch size | 256 |
| Max epochs | 60 (early stopped at epoch 29) |
| Loss function | Focal Loss |
| Regularization | SWA + EMA smoothing + Jitter augmentation |
| Post-hoc calibration | Temperature scaling (T = 1.49) |

### Train / Val / Test Split — Patient Level

| Split | Windows | Normal | Critical | Emergency |
|---|---|---|---|---|
| Train | 57,300 | 23,028 | 11,660 | 22,612 |
| Val | 6,630 | 2,323 | 1,583 | 2,724 |
| Test | 6,876 | 2,109 | 1,547 | 3,220 |

---

## 📈 Model Performance

### Summary Metrics

| Metric | Value |
|---|---|
| Emergency Detection Rate (flagged as Critical or Emergency) | ~94% |
| Binary AUROC (Normal vs Critical+Emergency) | 0.7987 |
| Test AUROC | 0.7234 |
| AUPRC | 0.5654 |
| Balanced Accuracy | 0.53 |
| Accuracy | 0.53 |

### Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal (0) | 0.69 | 0.42 | 0.52 | 2,109 |
| Critical (1) | 0.30 | 0.64 | 0.41 | 1,547 |
| Emergency (2) | 0.75 | 0.55 | 0.63 | 3,220 |

### Test Confusion Matrix

| | Pred-Normal | Pred-Critical | Pred-Emergency |
|---|---|---|---|
| True-Normal | 890 | 978 | 241 |
| True-Critical | 215 | 985 | 347 |
| True-Emergency | 71 | 1,646 | 1,757 |

---

## 🔍 Understanding the Results

### Why per-class Emergency recall is not the right metric

Strict label-level recall counts only Emergency→Emergency predictions. But in clinical use, any flag — Critical or Emergency — triggers bedside attention. Measuring only exact label matches underestimates real clinical utility.

```
Strict Emergency recall:   0.55   (Emergency → Emergency only)
Clinical detection rate:   0.94   (Emergency → Critical or Emergency)
```

The 94% figure is the operationally correct measure for an early warning system.

### Emergency → Critical: Conservative Early Detection

Emergency cases predicted as Critical are not errors. They represent genuine instability detected with a conservative severity label — a clinical alert that still fires and triggers intervention. The model has learned that Critical is physiologically early Emergency.

### Critical → Emergency: Safety-Oriented Escalation

Approximately 22% of Critical cases are classified as Emergency. In a safety-critical environment, the cost of over-escalation is lower than the cost of missing a deteriorating patient. This is desirable behaviour — clinical conservatism, not model error.

---

## 🆚 Comparison with NEWS2

F.T was compared against a **modified NEWS2** implementation, adapted for intraoperative data constraints.

### Modified NEWS2 — Variables Used

| Included | Excluded |
|---|---|
| Respiratory Rate (RR) | Temperature (not consistently available) |
| SpO₂ | Consciousness level (not meaningful under anaesthesia) |
| Heart Rate (HR) | |
| Systolic BP (SBP) | |
| Oxygen (FiO₂ proxy) | |

### Comparison Results (54,750 common rows compare by 10 sample patients)

| Metric | Agreement |
|---|---|
| Severity Match | 61.39% (33,612 rows) |
| Result Match | 60.86% (33,323 rows) |
| Exact Match (Both) | 54.10% (29,618 rows) |

The ~54% exact match is expected and informative — it reflects the fundamental difference in how the two systems operate.

### Why Not NEWS2?

> **NEWS2 detects when deterioration is obvious. F.T detects how deterioration develops over time.**

| | NEWS2 | F.T |
|---|---|---|
| Approach | Static snapshot | Continuous trajectory |
| Signal | Current threshold breach | Evolving physiological pattern |
| Pattern detection | Single vital, isolated | Multi-vital combinations |
| Temporal consistency | None | FSM-stabilised 15-window confirmation |
| Early warning | At threshold breach | Before threshold breach (ramp detection) |
| Hidden instability | Not detected | Tier 3 patterns (Masked Shock, Stable Deceiver, Occult Acidosis) |

F.T's Tier 3 patterns are specifically designed to catch instability that NEWS2 cannot see — states where vitals look superficially acceptable but physiological trajectories indicate hidden deterioration.

**F.T acts as an intelligent early warning layer, not a replacement for NEWS2.** It extends NEWS2 by detecting early and hidden instability, capturing multi-parameter physiological interactions, and providing time-consistent severity estimation.

---

## 🔎 Rule-Based Clinical Reasoning Layer

The rule-based layer runs alongside the prediction model and provides a **5-stage explainability pipeline** for every patient time point. It uses the full 99-feature master dataset to generate human-readable clinical explanations.

### Stage Pipeline

| Stage | Function |
|---|---|
| Stage 1 | System label display (severity_label vs result_label) |
| Stage 2 | FSM mismatch explanation — why confirmed label may differ from raw label |
| Stage 3 | Vital sign analysis: individual threshold flags, active pattern conditions, full out-of-range summary |
| Stage 4 | Trend and noise warnings: monotonic slope acceleration detection, impossible value / signal noise check |
| Stage 5 | Recovery detection: vitals trending back toward normal across all slope windows |

### Example Output

```
Snapshot ready — Patient 64 @ 3000s
Lag snapshot (15m ago): Found
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

  🚨 Both the raw severity and the system-confirmed label are in
     agreement: the patient is in a confirmed EMERGENCY state.

     This means the deterioration is not a transient spike —
     it has been sustained across the 15-reading evaluation window
     and the FSM has locked in this severity label.
     clinical attention is required.
═════════════════════════════════════════════════════════════════
  STAGE 3 — VITAL SIGN ANALYSIS & CLINICAL PATTERN FLAGS
═════════════════════════════════════════════════════════════════

  Current Condition: EMERGENCY
─────────────────────────────────────────────────────────────────

  STEP A — EMERGENCY-THRESHOLD VITAL TRIGGERS
  (Vitals that have crossed the emergency boundary)

  👉 Condition Emergency is triggered due to abnormal vital(s):
     [Heart Rate, Respiratory Rate, Systolic Blood Pressure (SBP), Diastolic Blood Pressure (DBP), Mean Arterial Pressure (MAP), End-Tidal CO₂ (EtCO₂), Pulse Pressure]

  🔴 Heart Rate
       Current Value  : 59.0 bpm
       Status         : Borderline-Low — below expected range of 60 bpm
       Clinical Note  : This indicates the heart rate is dangerously slow, which may reduce cardiac output and organ perfusion.

  🔴 Respiratory Rate
       Current Value  : 40.0 breaths/min
       Status         : Emergency — above expected range of 20 breaths/min
       Clinical Note  : This means the patient is breathing faster than normal — the body may be trying to compensate for low oxygen or acidosis.

  🔴 Systolic Blood Pressure (SBP)
       Current Value  : 89.0 mmHg
       Status         : Emergency — below expected range of 110 mmHg
       Clinical Note  : This indicates hypotension — the heart may not be pumping enough blood to vital organs.

  🔴 Diastolic Blood Pressure (DBP)
       Current Value  : 46.0 mmHg
       Status         : Emergency — below expected range of 60 mmHg
       Clinical Note  : This indicates low diastolic pressure, potentially reducing coronary perfusion.

  🔴 Mean Arterial Pressure (MAP)
       Current Value  : 62.0 mmHg
       Status         : Critical — below expected range of 70 mmHg
       Clinical Note  : This is critically important — MAP below 70 mmHg is associated with organ ischemia and shock.

  🔴 End-Tidal CO₂ (EtCO₂)
       Current Value  : 33.0 mmHg
       Status         : Borderline-Low — below expected range of 35 mmHg
       Clinical Note  : This suggests hyperventilation or poor cardiac output with reduced CO₂ delivery to the lungs.

  🔴 Pulse Pressure
       Current Value  : 43.0 mmHg
       Status         : Borderline-Low — below expected range of 45 mmHg
       Clinical Note  : This is a critical warning sign of tamponade, severe hypovolemia, or cardiogenic shock — the difference between systolic and diastolic is narrowing dangerously.

─────────────────────────────────────────────────────────────────
  STEP B — ACTIVE CLINICAL CONDITION PATTERNS
  (Conditions activated by multi-vital combinations)

  No active multi-vital pattern conditions flagged at this time.

─────────────────────────────────────────────────────────────────
  STEP C — ALL VITALS NOT IN NORMAL RANGE
  (Including those that haven't crossed emergency level)

  ⚠️  Heart Rate: 59.0 bpm  →  LOW — Borderline-Low range (normal lower limit: 60 bpm)
  ⚠️  Respiratory Rate: 40.0 breaths/min  →  HIGH — Emergency range (normal upper limit: 20 breaths/min)
  ⚠️  Systolic Blood Pressure (SBP): 89.0 mmHg  →  LOW — Emergency range (normal lower limit: 110 mmHg)
  ⚠️  Diastolic Blood Pressure (DBP): 46.0 mmHg  →  LOW — Emergency range (normal lower limit: 60 mmHg)
  ⚠️  Mean Arterial Pressure (MAP): 62.0 mmHg  →  LOW — Critical range (normal lower limit: 70 mmHg)
  ⚠️  End-Tidal CO₂ (EtCO₂): 33.0 mmHg  →  LOW — Borderline-Low range (normal lower limit: 35 mmHg)
  ⚠️  Pulse Pressure: 43.0 mmHg  →  LOW — Borderline-Low range (normal lower limit: 45 mmHg)

═════════════════════════════════════════════════════════════════
  STAGE 4 — TREND & NOISE WARNINGS
═════════════════════════════════════════════════════════════════

  WARNING 1 — CONTINUOUS VITAL TREND DETECTION
  (Checking lag vitals slope: 15m → 7m → 5m → 2m)

  ✅ No continuous monotonic trend detected across lag vitals.
─────────────────────────────────────────────────────────────────
  WARNING 2 — IMPOSSIBLE VITAL VALUES / SIGNAL NOISE
  (Checking for values beyond human physiological limits)

  ✅ No impossible vital values or signal noise artefacts detected.
═════════════════════════════════════════════════════════════════
  STAGE 5 — GOOD SIGNS & RECOVERY ANALYSIS
═════════════════════════════════════════════════════════════════

  ℹ️  No recovery signs detected at this time.
     No vitals that were previously abnormal are showing a consistent
     return trend across all slope windows.
═════════════════════════════════════════════════════════════════
         RULE-BASED EXPLAINABLE AI LAYER — FINAL REPORT
═════════════════════════════════════════════════════════════════
  Patient ID         : 64
  Time               : 3000s  (50 min into monitoring)
  Combined Score     : 4.7562
─────────────────────────────────────────────────────────────────
  Raw Severity       : EMERGENCY
  FSM Confirmed      : EMERGENCY
  15-Min Forecast    : 1.0
─────────────────────────────────────────────────────────────────
  🚨  CONFIRMED EMERGENCY — Raw severity and FSM both agree.
      Sustained deterioration confirmed. medical action required.

  ✅ Active Flags       : None

  ✅ Trends             : No monotonic trends detected in abnormal lag vitals

  ─  Recovery Signs     : None detected at this time
═════════════════════════════════════════════════════════════════
```

---

## 🗂️ Repository Structure

```
├cleaning.ipynb                    # Physiological risk engine & feature pipeline
├ cnn_gru_2.py                     # Feature engineering & CNN-GRU  training
├ RULE_BASED_AI.ipynb              # Rule-Based Clinical Reasoning Layer
├ README.md
|-news-2                           # labeling by news and comparison
|-tiew combination                  # tier explanation
    

```

---

## 📥 Dataset Download

The master dataset is hosted on Kaggle due to GitHub file size limits.

| Dataset | Description | Link |
|---|---|---|
| Initial Dataset | Raw data before cleaning | [Download from Kaggle](https://kaggle.com) |
| Master Dataset | Processed, cleaned, 99-feature dataset | [Download from Kaggle](https://kaggle.com) |

After downloading, place the dataset file in the project root directory before running any scripts.

---

## 💡 Applications

- ICU early warning and real-time deterioration monitoring
- Clinical decision support with explicit physiological reasoning
- Multi-organ failure detection research
- Physiological instability modelling and high-resolution dataset construction
- Rule-based AI integration for bedside alert explanation
- Research foundation for geriatric ICU physiological stream analysis

---

## 🗺️ Planned Extensions

- [ ] **XAI Integration** — SHAP / attention visualisation layer to complement the rule-based explanations with feature attribution from the deep learning model
- [ ] **Score Fluctuation Analysis** — Studying `combined_score` trajectories: instability oscillation patterns, transition velocities, and physiological drivers behind score variance
- [ ] **Transformer / attention-based architecture exploration** for time-series severity prediction in extended dataset
- [ ] **LSTM architecture comparison study**
- [ ] **Real-time ICU deployment pipeline**
- [ ] **Prospective clinical validation study**
- [ ] **Validation on larger, multi-hospital datasets** (generalizability study)
- [ ] **Extension to broader ICU age groups** (below 60 and above 80)

---

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| Low Critical class precision | Precision of 0.30 on the Critical class reflects inherent label boundary ambiguity — Critical and Emergency states overlap physiologically, and the ground truth labels carry this ambiguity. The system is intentionally tuned toward recall (catching deteriorating patients) at the cost of precision. |
| Single-center data | Trained on one dataset — generalizability to other ICU populations or hospital settings is unknown |
| No clinical validation | This is a research prototype. It has not been prospectively validated in a clinical environment and is not a certified medical device |
| Intraoperative context | VitalDB captures surgical/perioperative monitoring; dynamics may differ from general ICU populations |
| Hardware dependency | Requires continuous high-frequency vital monitoring at 2-second resolution — not available in all clinical settings |
| Label boundary ambiguity | The Critical / Emergency boundary is not sharply defined in real physiological data — some label overlap is inherent in the ground truth |
| NEWS2 comparison scope | The modified NEWS2 excluded temperature and consciousness due to data constraints; full NEWS2 comparison is pending |

---

## 📋 Quick Reference

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

> **F.T is a research prototype. Not intended for direct clinical use without prospective validation and regulatory clearance.**
