# F.T — Physiological Instability Detection Engine

A hybrid physiological reasoning and deep learning system for real-time ICU patient deterioration detection and prediction — up to 15 minutes in advance.

---

## What is F.T?

**F.T (Flow-Threshold)** is designed to detect and predict physiological deterioration in ICU patients aged 65 and above using continuous vital sign monitoring.

Rather than relying on simple alarm thresholds or pure machine learning, F.T combines four components:

- **Medical physiology rules** — clinically grounded deterioration patterns
- **Mathematical severity modeling** — continuous, nonlinear risk encoding
- **Temporal state logic** — FSM-based label stabilization
- **Deep learning prediction** — 1D CNN trained on engineered physiological trajectories

The result is a system that learns how deterioration unfolds over time, not just whether a value is abnormal at a single moment.

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
  · CatevCNN1D v8 (Multi-scale 1D CNN)
  · 3-class severity output: Normal / Critical / Emergency
```

---

## Dataset

**Source:** VitalDB ICU Dataset

| Property   | Value                          |
|------------|-------------------------------|
| Patients   | 103 ICU patients              |
| Age        | ≥ 65 years                   |
| Monitoring | Continuous vital sign streams |
| Resolution | 2-second intervals            |
| Total rows | ~620,000                      |

This forms a high-resolution geriatric ICU physiological stream dataset.

**Target distribution (`future_label`):**

| Class | Label     | Count   | Share  |
|-------|-----------|---------|--------|
| 0     | Normal    | 244,449 | 39.4%  |
| 1     | Critical  | 118,468 | 19.1%  |
| 2     | Emergency | 257,340 | 41.5%  |

---

## Vital Signals

### Primary Inputs

| Signal            | Column               | Description                        |
|-------------------|----------------------|------------------------------------|
| SpO₂              | `spo2`               | Oxygen saturation                  |
| Heart Rate        | `heart_rate`         | Pulse rate (bpm)                   |
| Respiratory Rate  | `resp_rate_smoothed` | RR with rolling smoothing applied  |
| Systolic BP       | `sbp`                | Systolic blood pressure            |
| Diastolic BP      | `dbp`                | Diastolic blood pressure           |
| End-Tidal CO₂     | `etco2`              | Ventilatory CO₂ marker             |

> Raw `resp_rate` is excluded from the model. Only the smoothed version `resp_rate_smoothed` is used to reduce sensor noise.

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

| Zone      | Meaning                    |
|-----------|---------------------------|
| Normal    | Physiologically stable    |
| Critical  | Significant abnormality   |
| Emergency | Severe instability        |

**Full threshold table:**

| Vital              | Normal       | Critical   | Emergency  |
|--------------------|-------------|------------|------------|
| SpO₂               | ≥ 95%       | 92–95%     | ≤ 90%      |
| HR (high)          | ≤ 90 bpm    | 90–110     | ≥ 120      |
| HR (low)           | ≥ 60 bpm    | 50–60      | ≤ 45       |
| RR (high)          | ≤ 20 /min   | 20–25      | ≥ 30       |
| RR (low)           | ≥ 12 /min   | 10–12      | ≤ 8        |
| SBP (low)          | ≥ 110 mmHg  | 100–110    | ≤ 90       |
| SBP (high)         | ≤ 150 mmHg  | 150–170    | ≥ 185      |
| DBP (low)          | ≥ 60 mmHg   | 55–60      | ≤ 50       |
| DBP (high)         | ≤ 85 mmHg   | 85–95      | ≥ 100      |
| MBP                | ≥ 70 mmHg   | 65–70      | ≤ 60       |
| ETCO₂ (high)       | ≤ 45 mmHg   | 45–50      | ≥ 55       |
| ETCO₂ (low)        | ≥ 35 mmHg   | 30–35      | ≤ 25       |
| Pulse Pressure (low) | ≥ 45 mmHg | 35–45      | ≤ 30       |
| Pulse Pressure (high)| ≤ 65 mmHg | 65–75      | ≥ 85       |

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

| z   | Severity |
|-----|----------|
| 0.0 | 0.00     |
| 0.5 | 0.41     |
| 1.0 | 1.00     |

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

| Pattern              | Trigger                        | Clinical Meaning                              |
|----------------------|-------------------------------|-----------------------------------------------|
| Shock Spiral         | MBP < 70 AND HR > 100         | Low perfusion with compensatory tachycardia  |
| Respiratory Burnout  | SpO₂ < 92 AND RR > 22         | Oxygen failure with increased respiratory effort |
| Hypercapnic Failure  | ETCO₂ > 50 AND RR < 10        | Ventilatory failure with CO₂ retention       |

### Tier 2 — Moderate Risk

| Pattern                        | Trigger                                      |
|-------------------------------|----------------------------------------------|
| Pulse Pressure Low             | Pulse Pressure ≤ 30                          |
| Wide PP + High SBP             | Pulse Pressure ≥ 70 AND SBP ≥ 170           |
| Respiratory-Hemodynamic Combo  | SpO₂ < 92 AND RR > 22 AND HR > 100          |

### Tier 3 — Subtle / Hidden Risk

| Pattern              | Trigger                                              |
|----------------------|------------------------------------------------------|
| Hypertensive Emergency | SBP ≥ 180 AND Pulse Pressure ≥ 70               |
| Stable Deceiver       | SpO₂ 92–94 AND HR 75–90 AND MBP 65–70           |
| Masked Shock          | MBP 65–72 AND HR < 90 (perfusion decline without tachycardia) |
| Occult Acidosis       | ETCO₂ ≤ 32 AND RR ≥ 24 AND SpO₂ 88–92          |
| Trend Decline         | Simultaneous adverse point-to-point changes in ETCO₂, SpO₂, HR |
| Trend Activate        | Slope-based sustained deterioration across 5–7 minute windows |

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

Where `A` is the condition activation strength (0–1) and multipliers are capped at **2.2** to prevent runaway escalation.

---

## Temporal Stability Engine

A Finite State Machine (FSM) prevents label flickering caused by noisy vital sign data.

**Key rules:**
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

### Raw Vitals (8)
`dbp · mbp · heart_rate · sbp · spo2 · etco2 · pulse_pressure · resp_rate_smoothed`

### Scaled Vitals (8)
Physiologically scaled versions of all 8 vitals:
`s_spo2 · s_hr · s_rr · s_sbp · s_dbp · s_mbp · s_etco2 · s_pp`

### Vital Slopes (16)
OLS slopes for all 8 vitals across 2 time windows:

| Window | Row Count | Trend Scope           |
|--------|-----------|-----------------------|
| 7m     | 210 rows  | Medium-term trend     |
| 15m    | 450 rows  | Sustained trajectory  |

Example columns: `slope_7m_spo2 · slope_15m_heart_rate · slope_7m_mbp · slope_15m_etco2`

### Combined Score Slopes (2)
`slope_7m_combined_score · slope_15m_combined_score`

### Rolling Statistics (6)
Computed over `combined_score`:

| Feature                  | Windows       |
|--------------------------|---------------|
| `roll_mean_{w}_combined` | 7m, 15m       |
| `roll_std_{w}_combined`  | 7m, 15m       |
| `roll_min_15m_combined`  | 15m only      |
| `roll_max_15m_combined`  | 15m only      |

### Physiological Instability Score (1)
`combined_score` — the output of the risk engine — used directly as a model feature.

---

## Deep Learning Model — CatevCNN1D v8

### Architecture

A **multi-scale parallel 1D CNN** with inception-style feature extraction:

```
Input: (batch, 45 timesteps, 41 features)
         ↓
  ┌──────────────────────────────────────┐
  │  Parallel Conv1D branches            │
  │  · Branch A: kernel=3, 64 filters    │
  │  · Branch B: kernel=5, 64 filters    │
  │  · Branch C: kernel=9, 64 filters    │
  │  Each followed by LayerNorm + Dropout│
  └──────────────┬───────────────────────┘
                 ↓ Concatenate → (45, 192)
          Conv1D 128 + LayerNorm + Dropout
                 ↓
          Conv1D 128 + LayerNorm + Dropout
                 ↓
         GlobalAveragePooling1D → (128,)
                 ↓
           Dense 64 + Dropout
                 ↓
        Output: Dense 3 (softmax)
```

**Total parameters:** 247,427 (~966 KB)

### Window Configuration

| Parameter | Value         | Description              |
|-----------|---------------|--------------------------|
| Window    | 45 rows (90s) | Input sequence length    |
| Stride    | 30 rows (60s) | Step between windows     |
| Overlap   | 33%           | Temporal continuity      |
| Jitter    | ±7 rows       | Augmentation during training |

### Training Configuration

| Parameter       | Value                    |
|----------------|--------------------------|
| Optimizer       | Adam, LR=0.0003         |
| Batch size      | 256                      |
| Max epochs      | 60 (early stopped at 11) |
| Loss            | Sparse categorical crossentropy |
| Class balancing | Balanced sample weights  |
| Scheduler       | ReduceLROnPlateau        |

### Train / Val / Test Split

Patient-level stratified split:

| Split | Patients | Class Distribution             |
|-------|----------|-------------------------------|
| Train | 80       | N=43.3%  C=20.2%  E=36.5%    |
| Val   | 10       | N=21.0%  C=21.6%  E=57.4%    |
| Test  | 13       | N=31.9%  C=10.4%  E=57.7%    |

**Windows generated:**
- Train: 30,230 windows
- Val: 2,228 windows
- Test: 2,664 windows

---

## Model Performance

### Validation Results

| Metric       | Value  |
|-------------|--------|
| Macro-F1    | 0.4671 |
| Bal. Acc.   | 0.4608 |
| Accuracy    | 0.54   |

| Class       | Precision | Recall | F1   | Support |
|-------------|-----------|--------|------|---------|
| Normal (0)  | 0.52      | 0.44   | 0.48 | 465     |
| Critical (1)| 0.26      | 0.26   | 0.26 | 490     |
| Emergency (2)| 0.65     | 0.69   | 0.67 | 1273    |

**Validation confusion matrix (normalised):**

|                  | Pred-Normal | Pred-Critical | Pred-Emergency |
|------------------|-------------|---------------|----------------|
| True-Normal      | 44.1%       | 13.1%         | 42.8%          |
| True-Critical    | 17.4%       | 25.5%         | 57.1%          |
| True-Emergency   | 7.9%        | 23.4%         | 68.7%          |

### Test Results

| Metric       | Value  |
|-------------|--------|
| Macro-F1    | 0.6053 |
| Bal. Acc.   | 0.6144 |
| Accuracy    | 0.72   |

| Class        | Precision | Recall | F1   | Support |
|--------------|-----------|--------|------|---------|
| Normal (0)   | 0.83      | 0.86   | 0.84 | 852     |
| Critical (1) | 0.16      | 0.27   | 0.20 | 281     |
| Emergency (2)| 0.84      | 0.72   | 0.77 | 1531    |

**Test confusion matrix (normalised):**

|                  | Pred-Normal | Pred-Critical | Pred-Emergency |
|------------------|-------------|---------------|----------------|
| True-Normal      | 85.9%       | 5.5%          | 8.6%           |
| True-Critical    | 23.5%       | 26.7%         | 49.8%          |
| True-Emergency   | 5.6%        | 22.7%         | 71.7%          |

> **Note:** Critical class remains the hardest to classify — a known challenge given its transitional physiological nature between Normal and Emergency states.

### Top Predictive Features

| Rank | Feature                  | Interpretation                        |
|------|--------------------------|---------------------------------------|
| 1    | `roll_max_15m_combined`  | Peak instability over 15 minutes      |
| 2    | `roll_std_15m_combined`  | Volatility of instability score       |
| 3    | `roll_mean_15m_combined` | Sustained average instability         |
| 4    | `slope_15m_etco2`        | CO₂ trend — ventilatory trajectory   |
| 5    | `slope_15m_heart_rate`   | HR trend — cardiac trajectory         |
| 6    | `pulse_pressure`         | Vascular instability marker           |

All top features represent **physiological trajectories**, not isolated abnormal values.

---

## Repository Structure

```
├── catevcode.py                  # Physiological risk engine & feature pipeline
├── catev_cnn_v8_training.py      # Feature engineering & CNN training
└── README.md
```

**Saved model artifacts:**
```
catev_cnn_v8_model.keras          # Trained model weights
catev_cnn_v8_meta.pkl             # Scaler and metadata
catev_cnn_v8_report.txt           # Full evaluation report
catev_cnn_v8_history.png          # Training curves
catev_cnn_v8_validation_cm.png    # Validation confusion matrix
catev_cnn_v8_test_cm.png          # Test confusion matrix
```

---

## Applications

- ICU early warning and real-time deterioration monitoring
- Clinical decision support for bedside staff
- Multi-organ failure detection research
- Physiological instability modeling and dataset construction

---

## Limitations

- Single-center dataset — generalizability to other ICU populations is unknown
- Critical class classification remains challenging due to its transitional physiological nature
- Currently a research prototype, not a certified clinical product
- Requires continuous high-frequency vital monitoring at 2-second resolution
- Val/Test class distributions differ from training — reflects real patient variability but limits direct comparison

---

## Planned Extensions

- [ ] **Rule-Based AI Layer** — A structured reasoning layer enabling explicit clinical logic to interpret and explain instability classifications without relying solely on learned patterns
- [ ] **Score Fluctuation Analysis** — Studying `combined_score` trajectories: instability oscillation patterns, transition velocities, and physiological drivers behind score variance
- [ ] Validation on larger, multi-hospital datasets
- [ ] Transformer / LSTM time-series models
- [ ] Real-time ICU deployment pipeline
- [ ] Prospective clinical validation study
- [ ] Extension to broader ICU age groups

---

## Dataset

The master dataset used for this project is hosted on Kaggle due to GitHub file size limits.

📥 **[Download the dataset from Kaggle](https://www.kaggle.com/datasets/arjunmahesh09999/final)**

After downloading, place the dataset file inside the project directory before running the code.

---

## Author

**Arjun**  
MSc Artificial Intelligence & Machine Learning  
Indian Institute of Information Technology, Lucknow (IIIT-L)
