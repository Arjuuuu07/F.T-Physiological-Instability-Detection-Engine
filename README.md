# F.T — Physiological Instability Detection Engine

> A hybrid physiological reasoning and machine learning system for real-time ICU patient deterioration detection.

---

## What is F.T?

**F.T (Flow-Threshold)** combines clinical physiological modeling with machine learning to detect and predict deterioration in ICU patients. Rather than relying on simple alarm thresholds, F.T learns the *trajectory* of physiological instability — catching deterioration before it becomes serious.

---

## System Pipeline

```
Raw ICU Vital Streams
        ↓
Physiological Risk Engine
        ↓
Disease Pattern Detection
        ↓
Temporal Stability Modeling (FSM)
        ↓
Feature Engineering
        ↓
Machine Learning Prediction
```

Each stage progressively transforms raw time-series data into a structured instability signal for the ML model.

---

## Dataset

**Source:** [VitalDB ICU Dataset](https://vitaldb.net/)

| Property | Detail |
|---|---|
| Patients | 31 ICU patients |
| Age | ≥ 65 years |
| Time rows | ~211,000 |
| Resolution | 2-second intervals |

This forms a high-resolution geriatric ICU physiological stream dataset.

---

## Vital Signals

**Primary signals monitored:**

| Signal | Description |
|---|---|
| SpO₂ | Oxygen saturation |
| HR | Heart rate |
| RR | Respiratory rate |
| SBP / DBP | Systolic / Diastolic blood pressure |
| ETCO₂ | End-tidal CO₂ |

**Derived signals:**

```
Pulse Pressure  =  SBP − DBP
MBP             =  (SBP + 2 × DBP) / 3
```

> Respiratory rate is smoothed using rolling averages to reduce signal noise.

---

## Physiological Risk Engine

Raw vitals are converted into a continuous physiological instability score using three components.

### 1. Threshold Zones

Each vital is divided into severity zones:

| Zone | Meaning |
|---|---|
| Normal | Stable physiology |
| Critical | Moderate instability |
| Emergency | Severe deterioration |

**Example thresholds:**

| Vital | Normal | Critical | Emergency |
|---|---|---|---|
| SpO₂ | ≥ 95% | 92% | 90% |
| HR (high) | 90 bpm | 110 bpm | 120 bpm |
| MBP (low) | 70 mmHg | 65 mmHg | 60 mmHg |
| RR (high) | 20 /min | 25 /min | 30 /min |
| ETCO₂ (high) | 45 mmHg | 50 mmHg | 55 mmHg |

### 2. Continuous Abnormality Encoding

Each vital is mapped to a continuous score `z ∈ [0, 1]`:

```
0.0  →  Normal
0.5  →  Critical boundary
1.0  →  Emergency boundary
```

This avoids abrupt threshold jumps and models gradual deterioration.

### 3. Nonlinear Severity Escalation

```
s = 2z² − 1
```

Severe abnormalities escalate rapidly in contribution, reflecting real clinical urgency.

### 4. Multi-Organ Risk Aggregation

```
severity_sum = Σ sᵢ  (across all vital signs)
```

Models cumulative multi-organ physiological stress.

---

## Disease Pattern Modeling

F.T encodes clinically meaningful deterioration patterns across three tiers.

### Tier 1 — High Risk

| Pattern | Trigger |
|---|---|
| **Shock Spiral** | MBP < 70 AND HR > 100 |
| **Respiratory Burnout** | SpO₂ < 92% AND RR > 22 |
| **Hypercapnic Failure** | ETCO₂ > 50 AND RR < 10 |

### Tier 2 — Moderate Risk

- Low pulse pressure
- Wide pulse pressure with high SBP
- Respiratory-hemodynamic interaction

### Tier 3 — Subtle Physiological Stress

- Masked shock
- Occult metabolic instability
- Hidden deterioration trends

### Early Warning Ramp

Deterioration detection begins *before* thresholds are crossed:

```
early_start = threshold ± 0.2 × (threshold − normal_reference)
```

When one variable deteriorates, ramp thresholds for related vitals shrink by up to **50%**, modeling multi-organ failure cascade dynamics.

---

## Temporal Stability Engine

A **Finite State Machine (FSM)** prevents label flickering from noisy data.

Key rules:
- **15 consecutive identical states** are required to confirm a label
- **Emergency → Normal** direct transition is not permitted
- Mixed Critical/Emergency states collapse to **Critical**
- Downgrades require **sustained recovery** over time

This ensures physiological state transitions are clinically meaningful.

---

## Final Instability Score

```
final_score = severity_sum × M_eff

M_eff = 1 + A(target − 1)
```

Tier multipliers adjust severity based on active disease patterns.

### Classification

| Score | Label |
|---|---|
| < 0.5 | ✅ Normal |
| ≥ 0.5 | ⚠️ Critical |
| ≥ 1.2 | 🚨 Emergency |

---

## Feature Engineering

Temporal deterioration patterns are captured through a rich feature set:

| Feature Type | Examples |
|---|---|
| Raw vitals | SpO₂, HR, RR, SBP, DBP, ETCO₂ |
| Vital slopes | `slope_2m_heart_rate`, `slope_15m_etco2` |
| Rolling statistics | `roll_mean_15m_combined`, `roll_std_7m_combined` |
| Lagged vitals | `lag_15m_pulse_pressure` |
| Instability dynamics | `roll_max_15m_combined` |

Slopes are computed across **2m, 5m, 7m, and 15m** windows to capture short- and medium-term trends.

---

## Machine Learning

**Primary model:** `HistGradientBoostingClassifier`

Models evaluated: HistGradientBoosting, RandomForest — selected by **macro-F1 score**.

### Performance (5-Fold Cross-Validation)

| Metric | Score |
|---|---|
| Macro F1 | **0.958** |
| Balanced Accuracy | **0.962** |

### Top Predictive Features

1. `roll_max_15m_combined`
2. `roll_std_15m_combined`
3. `roll_mean_15m_combined`
4. `slope_15m_etco2`
5. `slope_15m_heart_rate`
6. `pulse_pressure`

These features represent physiological instability *trajectories*, not isolated abnormal values.

---

## Applications

- ICU early warning systems
- Real-time patient deterioration monitoring
- Clinical decision support tools
- Multi-organ failure detection research
- Physiological instability modeling

---

## Limitations

- Small patient cohort (31 patients) — external validation required
- Currently a **research prototype**, not a clinical product
- Single-center dataset — generalizability to be assessed

---

## Future Work

- [ ] Larger, multi-hospital datasets
- [ ] Deep learning time-series models (e.g., Transformers, LSTMs)
- [ ] Real-time ICU deployment pipeline
- [ ] Prospective clinical validation study

---

## Author

**Arjun**
MSc Artificial Intelligence & Machine Learning
*Machine Learning & Physiological Modeling Research*
