# F.T — Physiological Instability Detection Engine

> A hybrid physiological reasoning and machine learning system for real-time ICU patient deterioration detection and prediction — up to **15 minutes in advance**.

---

## What is F.T?

**F.T (Flow-Threshold)** is designed to detect and predict physiological deterioration in ICU patients aged 65 and above using continuous vital sign monitoring.

Rather than relying on simple alarm thresholds or pure machine learning, F.T combines four components:

- **Medical physiology rules** — clinically grounded deterioration patterns
- **Mathematical severity modeling** — continuous, nonlinear risk encoding
- **Temporal state logic** — FSM-based label stabilization
- **Machine learning prediction** — trained on engineered physiological trajectories

The result is a system that learns *how deterioration unfolds over time*, not just whether a value is abnormal at a single moment.

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
  · Synergistic physiological interaction
  · Condition amplification multiplier
          ↓
Temporal Stability Modeling (FSM)
  · Label confirmation & state transition rules
          ↓
Feature Engineering
  · Raw vitals · Slopes · Rolling stats · Lags · Condition flags
          ↓
Machine Learning Prediction
  · HistGradientBoostingClassifier
  · 3-class severity output: Normal / Critical / Emergency
```

---

## Dataset

**Source:** [VitalDB ICU Dataset](https://vitaldb.net/)

| Property | Value |
|---|---|
| Patients | 31 ICU patients |
| Age | ≥ 65 years |
| Monitoring | Continuous vital sign streams |
| Resolution | 2-second intervals |
| Total rows | ~211,000 |

This forms a high-resolution geriatric ICU physiological stream dataset.

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

> Raw `resp_rate` is **excluded** from the model. Only the smoothed version `resp_rate_smoothed` is used to reduce sensor noise.

### Derived Signals (treated as first-class features)

```
Pulse Pressure  =  SBP − DBP
MBP             =  (SBP + 2 × DBP) / 3
```

`pulse_pressure` is computed from raw inputs. `mbp` is already present in the dataset and used directly as a feature.

---

## Physiological Risk Engine

### Step 1 — Threshold Zone Mapping

Each vital is divided into three clinical risk zones:

| Zone | Meaning |
|---|---|
| Normal | Physiologically stable |
| Critical | Significant abnormality |
| Emergency | Severe instability |

Full threshold table:

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

Each z-score is transformed to emphasize extreme abnormalities:

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

F.T encodes 12 clinically meaningful deterioration patterns across three tiers.

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

| Pattern | Trigger |
|---|---|
| **Hypertensive Emergency** | SBP ≥ 180 AND Pulse Pressure ≥ 70 |
| **Stable Deceiver** | SpO₂ 92–94 AND HR 75–90 AND MBP 65–70 |
| **Masked Shock** | MBP 65–72 AND HR < 90 (perfusion decline without tachycardia) |
| **Occult Acidosis** | ETCO₂ ≤ 32 AND RR ≥ 24 AND SpO₂ 88–92 |
| **Trend Decline** | Simultaneous adverse point-to-point changes in ETCO₂, SpO₂, HR |
| **Trend Activate** | Slope-based sustained deterioration across 5–7 minute windows |

### Early Deterioration Ramp

Detection begins *before* thresholds are crossed:

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

Where `A` is the condition activation strength (0–1) and multipliers are capped at **2.2** to prevent runaway escalation.

---

## Temporal Stability Engine

A **Finite State Machine (FSM)** prevents label flickering caused by noisy vital sign data.

Key rules:

- **15 consecutive identical states** required to confirm a label change
- **Emergency → Normal** direct transition is blocked
- Mixed Critical / Emergency states collapse to **Critical**
- Downgrades require **sustained recovery** — not a single normal reading

This ensures state transitions reflect genuine physiological change, not sensor artifacts.

---

## Severity Classification

```
final_score < 0.75            →  ✅ Normal
0.75 ≤ final_score < 1.5      →  ⚠️  Critical
final_score ≥ 1.5             →  🚨 Emergency
```

---

## Feature Engineering

Temporal deterioration patterns are captured through **~68 engineered features** across six categories.

### Raw Vitals (8)

`spo2` · `heart_rate` · `resp_rate_smoothed` · `sbp` · `dbp` · `mbp` · `etco2` · `pulse_pressure`

### Vital Slopes (36)

OLS slopes computed for all 8 vitals + `combined_score` across 4 time windows:

| Window | Row Count | Trend Scope |
|---|---|---|
| 2m | 60 rows | Short-term change |
| 5m | 150 rows | Medium-term trend |
| 7m | 210 rows | Medium-term trend |
| 15m | 450 rows | Sustained trajectory |

Example columns: `slope_2m_spo2` · `slope_5m_heart_rate` · `slope_7m_mbp` · `slope_15m_etco2`

### Rolling Statistics (10)

Computed over `combined_score`:

| Feature | Windows |
|---|---|
| `roll_mean_{w}_combined` | 2m, 5m, 7m, 15m |
| `roll_std_{w}_combined` | 2m, 5m, 7m, 15m |
| `roll_min_15m_combined` | 15m only |
| `roll_max_15m_combined` | 15m only |

### Lag Features (9)

15-minute lookback (450 rows) for all 8 vitals + `combined_score`:

`lag_15m_spo2` · `lag_15m_heart_rate` · `lag_15m_etco2` · `lag_15m_pulse_pressure` · `lag_15m_combined_score` · ...

### Condition Binary Flags (12)

One binary flag per disease pattern — Tier 1 (3 flags), Tier 2 (3 flags), Tier 3 (6 flags).

### Physiological Instability Score (1)

`combined_score` — the output of the risk engine — used directly as a model feature.

---

## Machine Learning

**Primary model:** `HistGradientBoostingClassifier`

**Models evaluated:** HistGradientBoosting, RandomForest — winner selected by **macro-F1 score**.

**Split strategy:** Row-wise stratified 80/20 hold-out. Each row encodes its own temporal context via rolling and lag features, making row-wise splitting valid.

**Class balancing:** Balanced sample weights computed from training rows only, applied during `fit()`.

### Cross-Validation Performance (5-Fold, StratifiedKFold)

| Metric | Score |
|---|---|
| Macro F1 | **0.958 ± 0.004** |
| Balanced Accuracy | **0.962 ± 0.003** |

Hold-out test performance is consistent with CV results.

### Top Predictive Features

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `roll_max_15m_combined` | Peak instability over 15 minutes |
| 2 | `roll_std_15m_combined` | Volatility of instability score |
| 3 | `roll_mean_15m_combined` | Sustained average instability |
| 4 | `slope_15m_etco2` | CO₂ trend — ventilatory trajectory |
| 5 | `slope_15m_heart_rate` | HR trend — cardiac trajectory |
| 6 | `pulse_pressure` | Vascular instability marker |

All top features represent physiological *trajectories*, not isolated abnormal values.

---

## Repository Structure

```
├── catevcode.py                  # Physiological risk engine & feature pipeline
├── catev_model_v2_training.py    # Feature engineering & ML training
└── README.md
```

---

## Applications

- ICU early warning and real-time deterioration monitoring
- Clinical decision support for bedside staff
- Multi-organ failure detection research
- Physiological instability modeling and dataset construction

---

## Limitations

- Small patient cohort (31 patients) — external validation required
- Single-center dataset — generalizability to other ICU populations is unknown
- Currently a **research prototype**, not a certified clinical product
- Requires continuous high-frequency vital monitoring at 2-second resolution

---

## Planned Extensions

- [ ] **Rule-Based AI Layer** — A structured reasoning layer built on top of the existing physiological engine, enabling explicit clinical logic to interpret and explain instability classifications without relying solely on learned patterns.
- [ ] **Score Fluctuation Analysis** — Using the existing dataset to study how `combined_score` fluctuates across patient trajectories: identifying instability oscillation patterns, transition velocities between severity states, and the physiological drivers behind score variance.
- [ ] Validation on larger, multi-hospital datasets
- [ ] Deep learning time-series models (Transformers, LSTMs)
- [ ] Real-time ICU deployment pipeline
- [ ] Prospective clinical validation study
- [ ] Extension to broader ICU age groups

---

## Author

**Arjun**
MSc Artificial Intelligence & Machine Learning
Indian Institute of Information Technology, Lucknow (IIIT-L)
