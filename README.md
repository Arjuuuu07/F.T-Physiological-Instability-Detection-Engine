F.T — Physiological Instability Detection Engine

F.T (Flow-Threshold) is a hybrid physiological reasoning and machine learning system designed to detect and predict clinical deterioration in ICU patients using high-frequency vital sign data.

The system integrates deterministic physiological modeling, disease-pattern logic, temporal state stabilization, and machine learning to identify patient instability trajectories in real-time monitoring streams.

Overview

ICU monitoring systems continuously generate large volumes of physiological data. Traditional early-warning systems rely on simple threshold tables and often fail to capture the dynamic progression of physiological deterioration.

F.T addresses this limitation by combining clinical reasoning with machine learning.

System pipeline:

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

The model learns temporal physiological instability patterns rather than isolated abnormal values.

Dataset

Source: VitalDB ICU Dataset

Population selection rules:

Age ≥ 65 years

ICU monitored patients

Continuous physiological monitoring

2-second resolution time series

Dataset scale:

~211,000 time rows

31 ICU patients

High-frequency vital monitoring

This forms a high-resolution geriatric ICU physiological stream dataset.

Vital Signals Used

Primary physiological signals:

SpO₂ (oxygen saturation)

Heart rate

Respiratory rate

SBP (systolic blood pressure)

DBP (diastolic blood pressure)

ETCO₂ (end-tidal CO₂)

Derived signals:

Pulse Pressure

Pulse Pressure = SBP − DBP

Mean Blood Pressure

MBP = (SBP + 2 × DBP) / 3

Respiratory rate is smoothed using rolling averages to reduce noise.

Physiological Risk Engine

Raw vitals are converted into a continuous physiological instability representation.

Vital Threshold Zones

Each vital sign is divided into three zones:

Zone	Meaning
Normal	Stable physiology
Critical	Moderate instability
Emergency	Severe deterioration

Example thresholds:

Vital	Normal	Critical	Emergency
SpO₂	≥95	92	90
HR (high)	90	110	120
MBP (low)	70	65	60
RR (high)	20	25	30
ETCO₂ (high)	45	50	55
Continuous Abnormality Encoding

Each vital value is mapped to a continuous abnormality score:

z ∈ [0,1]

Where:

0   → Normal
0.5 → Critical boundary
1   → Emergency boundary

This avoids abrupt threshold transitions.

Nonlinear Severity Escalation

Severity contribution of each vital is modeled as:

s = 2z² − 1

This produces nonlinear escalation, where severe abnormalities grow rapidly.

Multi-Organ Risk Aggregation

Total physiological risk is calculated as:

severity_sum = Σ s_i

This models cumulative multi-organ stress.

Disease Pattern Modeling

F.T includes clinically meaningful deterioration patterns.

Tier 1 (High Risk)

Shock Spiral

MBP < 70 AND HR > 100

Respiratory Burnout

SpO₂ < 92 AND RR > 22

Hypercapnic Failure

ETCO₂ > 50 AND RR < 10
Tier 2 (Moderate Risk)

Low pulse pressure

Wide pulse pressure with high SBP

Respiratory-hemodynamic interaction

Tier 3 (Subtle Physiological Stress)

Masked shock

Occult metabolic instability

Hidden deterioration trends

Early Warning Ramp Activation

Risk activation begins before thresholds are crossed.

Ramp start:

early_start = threshold ± 0.2 × (threshold − normal_reference)

This allows early deterioration detection.

Synergistic Physiological Interaction

When one physiological variable deteriorates, others become more sensitive.

Ramp thresholds shrink by up to 50%, modeling multi-organ failure cascades.

Temporal Stability Engine

A Finite State Machine (FSM) prevents label flickering.

Rules include:

15 consecutive identical states confirm a label

Emergency cannot transition directly to Normal

Mixed Critical/Emergency collapses to Critical

Downgrades require sustained recovery

This stabilizes physiological state transitions.

Final Instability Score

The final physiological instability score is:

final_score = severity_sum × M_eff

Where

M_eff = 1 + A(target − 1)

Tier multipliers adjust severity based on disease pattern activation.

Severity Classification
Score	Label
< 0.5	Normal
≥ 0.5	Critical
≥ 1.2	Emergency
Feature Engineering

Temporal deterioration patterns are captured through engineered features.

Features include:

Raw vitals

Vital slopes (2m, 5m, 7m, 15m)

Rolling statistics

Lagged vitals (15-minute history)

Instability score dynamics

Example features:

slope_15m_heart_rate
roll_mean_15m_combined
lag_15m_pulse_pressure
roll_std_7m_combined
Machine Learning Model

Primary model:

HistGradientBoostingClassifier

Models evaluated:

HistGradientBoosting

RandomForest

Model selection based on macro-F1 score.

Model Performance

Cross-validation (5-fold):

Macro F1 Score: 0.958
Balanced Accuracy: 0.962

Hold-out test performance remains consistent.

Key Predictive Signals

Top feature contributors include:

roll_max_15m_combined

roll_std_15m_combined

roll_mean_15m_combined

slope_15m_etco2

slope_15m_heart_rate

pulse_pressure

These represent physiological instability trajectories.

Applications

Potential applications:

ICU early warning systems

Patient deterioration monitoring

Clinical decision support

Physiological instability research

Multi-organ failure detection

Limitations

Limited patient population (31 patients)

Requires external clinical validation

Currently a research prototype

Future Work

Larger multi-hospital datasets

Deep learning time-series models

Real-time ICU deployment

Prospective clinical validation

Author

Arjun
MSc Ai&ml
Machine Learning & Physiological Modeling Research
