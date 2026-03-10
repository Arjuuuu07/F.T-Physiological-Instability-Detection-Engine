F.T — Physiological Instability Detection Engine
A hybrid physiological reasoning and deep learning system for real-time ICU patient deterioration detection and prediction — up to 15 minutes in advance.

What is F.T?
F.T (Flow-Threshold) is designed to detect and predict physiological deterioration in ICU patients aged 65 and above using continuous vital sign monitoring.
Rather than relying on simple alarm thresholds or pure machine learning, F.T combines four components:

Medical physiology rules — clinically grounded deterioration patterns
Mathematical severity modeling — continuous, nonlinear risk encoding
Temporal state logic — FSM-based label stabilization
Deep learning prediction — 1D CNN trained on engineered physiological trajectories

The result is a system that learns how deterioration unfolds over time, not just whether a value is abnormal at a single moment.

System Pipeline
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

Dataset
Source: VitalDB ICU Dataset
PropertyValuePatients103 ICU patientsAge≥ 65 yearsMonitoringContinuous vital sign streamsResolution2-second intervalsTotal rows~620,000
This forms a high-resolution geriatric ICU physiological stream dataset.
Target distribution (future_label):
ClassLabelCountShare0Normal244,44939.4%1Critical118,46819.1%2Emergency257,34041.5%

Vital Signals
Primary Inputs
SignalColumnDescriptionSpO₂spo2Oxygen saturationHeart Rateheart_ratePulse rate (bpm)Respiratory Rateresp_rate_smoothedRR with rolling smoothing appliedSystolic BPsbpSystolic blood pressureDiastolic BPdbpDiastolic blood pressureEnd-Tidal CO₂etco2Ventilatory CO₂ marker

Raw resp_rate is excluded from the model. Only the smoothed version resp_rate_smoothed is used to reduce sensor noise.

Derived Signals (treated as first-class features)
Pulse Pressure  =  SBP − DBP
MBP             =  (SBP + 2 × DBP) / 3
pulse_pressure is computed from raw inputs. mbp is already present in the dataset and used directly as a feature.

Physiological Risk Engine
Step 1 — Threshold Zone Mapping
Each vital is divided into three clinical risk zones:
ZoneMeaningNormalPhysiologically stableCriticalSignificant abnormalityEmergencySevere instability
Full threshold table:
VitalNormalCriticalEmergencySpO₂≥ 95%92–95%≤ 90%HR (high)≤ 90 bpm90–110≥ 120HR (low)≥ 60 bpm50–60≤ 45RR (high)≤ 20 /min20–25≥ 30RR (low)≥ 12 /min10–12≤ 8SBP (low)≥ 110 mmHg100–110≤ 90SBP (high)≤ 150 mmHg150–170≥ 185DBP (low)≥ 60 mmHg55–60≤ 50DBP (high)≤ 85 mmHg85–95≥ 100MBP≥ 70 mmHg65–70≤ 60ETCO₂ (high)≤ 45 mmHg45–50≥ 55ETCO₂ (low)≥ 35 mmHg30–35≤ 25Pulse Pressure (low)≥ 45 mmHg35–45≤ 30Pulse Pressure (high)≤ 65 mmHg65–75≥ 85
Step 2 — Continuous Abnormality Encoding
Rather than binary zone membership, each vital is mapped to a continuous score z ∈ [0, 1]:
z = 0.0  →  Normal (no abnormality)
z = 0.5  →  Critical boundary
z = 1.0  →  Emergency boundary
This models gradual physiological deterioration rather than abrupt threshold jumps.
Step 3 — Nonlinear Severity Transformation
Each z-score is transformed to emphasize extreme abnormalities:
severity = 2^z − 1
zSeverity0.00.000.50.411.01.00
Severity grows faster near emergency levels, reflecting the nonlinear escalation of clinical risk.
Step 4 — Multi-Organ Risk Aggregation
severity_sum = Σ severity_i  (across all 8 vital signs)
This captures both single severe abnormalities and multiple concurrent mild abnormalities — modeling cumulative multi-organ physiological stress.

Disease Pattern Modeling
F.T encodes 12 clinically meaningful deterioration patterns across three tiers.
Tier 1 — Major Instability
PatternTriggerClinical MeaningShock SpiralMBP < 70 AND HR > 100Low perfusion with compensatory tachycardiaRespiratory BurnoutSpO₂ < 92 AND RR > 22Oxygen failure with increased respiratory effortHypercapnic FailureETCO₂ > 50 AND RR < 10Ventilatory failure with CO₂ retention
Tier 2 — Moderate Risk
PatternTriggerPulse Pressure LowPulse Pressure ≤ 30Wide PP + High SBPPulse Pressure ≥ 70 AND SBP ≥ 170Respiratory-Hemodynamic ComboSpO₂ < 92 AND RR > 22 AND HR > 100
Tier 3 — Subtle / Hidden Risk
PatternTriggerHypertensive EmergencySBP ≥ 180 AND Pulse Pressure ≥ 70Stable DeceiverSpO₂ 92–94 AND HR 75–90 AND MBP 65–70Masked ShockMBP 65–72 AND HR < 90 (perfusion decline without tachycardia)Occult AcidosisETCO₂ ≤ 32 AND RR ≥ 24 AND SpO₂ 88–92Trend DeclineSimultaneous adverse point-to-point changes in ETCO₂, SpO₂, HRTrend ActivateSlope-based sustained deterioration across 5–7 minute windows
Early Deterioration Ramp
Detection begins before thresholds are crossed:
early_start = threshold − 20% × (threshold − normal_reference)
This allows warning signals to develop before full clinical failure.
Condition Amplification
Active conditions amplify the final instability score:
final_score = severity_sum × M_eff
M_eff = 1 + A × (target_multiplier − 1)
Where A is the condition activation strength (0–1) and multipliers are capped at 2.2 to prevent runaway escalation.

Temporal Stability Engine
A Finite State Machine (FSM) prevents label flickering caused by noisy vital sign data.
Key rules:

15 consecutive identical states required to confirm a label change
Emergency → Normal direct transition is blocked
Mixed Critical / Emergency states collapse to Critical
Downgrades require sustained recovery — not a single normal reading

This ensures state transitions reflect genuine physiological change, not sensor artifacts.
Severity Classification
final_score < 0.75            →  ✅ Normal
0.75 ≤ final_score < 1.5      →  ⚠️  Critical
final_score ≥ 1.5             →  🚨 Emergency

Feature Engineering
Temporal deterioration patterns are captured through 41 engineered features across six categories.
Raw Vitals (8)
dbp · mbp · heart_rate · sbp · spo2 · etco2 · pulse_pressure · resp_rate_smoothed
Scaled Vitals (8)
Physiologically scaled versions of all 8 vitals:
s_spo2 · s_hr · s_rr · s_sbp · s_dbp · s_mbp · s_etco2 · s_pp
Vital Slopes (16)
OLS slopes for all 8 vitals across 2 time windows:
WindowRow CountTrend Scope7m210 rowsMedium-term trend15m450 rowsSustained trajectory
Example columns: slope_7m_spo2 · slope_15m_heart_rate · slope_7m_mbp · slope_15m_etco2
Combined Score Slopes (2)
slope_7m_combined_score · slope_15m_combined_score
Rolling Statistics (6)
Computed over combined_score:
FeatureWindowsroll_mean_{w}_combined7m, 15mroll_std_{w}_combined7m, 15mroll_min_15m_combined15m onlyroll_max_15m_combined15m only
Physiological Instability Score (1)
combined_score — the output of the risk engine — used directly as a model feature.

Deep Learning Model — CatevCNN1D v8
Architecture
A multi-scale parallel 1D CNN with inception-style feature extraction:
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
Total parameters: 247,427 (~966 KB)
Window Configuration
ParameterValueDescriptionWindow45 rows (90s)Input sequence lengthStride30 rows (60s)Step between windowsOverlap33%Temporal continuityJitter±7 rowsAugmentation during training
Training Configuration
ParameterValueOptimizerAdam, LR=0.0003Batch size256Max epochs60 (early stopped at 11)LossSparse categorical crossentropyClass balancingBalanced sample weightsSchedulerReduceLROnPlateau
Train / Val / Test Split
Patient-level stratified split:
SplitPatientsClass DistributionTrain80N=43.3%  C=20.2%  E=36.5%Val10N=21.0%  C=21.6%  E=57.4%Test13N=31.9%  C=10.4%  E=57.7%
Windows generated:

Train: 30,230 windows
Val: 2,228 windows
Test: 2,664 windows


Model Performance
Validation Results
MetricValueMacro-F10.4671Bal. Acc.0.4608Accuracy0.54
ClassPrecisionRecallF1SupportNormal (0)0.520.440.48465Critical (1)0.260.260.26490Emergency (2)0.650.690.671273
Validation confusion matrix (normalised):
Pred-NormalPred-CriticalPred-EmergencyTrue-Normal44.1%13.1%42.8%True-Critical17.4%25.5%57.1%True-Emergency7.9%23.4%68.7%
Test Results
MetricValueMacro-F10.6053Bal. Acc.0.6144Accuracy0.72
ClassPrecisionRecallF1SupportNormal (0)0.830.860.84852Critical (1)0.160.270.20281Emergency (2)0.840.720.771531
Test confusion matrix (normalised):
Pred-NormalPred-CriticalPred-EmergencyTrue-Normal85.9%5.5%8.6%True-Critical23.5%26.7%49.8%True-Emergency5.6%22.7%71.7%

Note: Critical class remains the hardest to classify — a known challenge given its transitional physiological nature between Normal and Emergency states.

Top Predictive Features
RankFeatureInterpretation1roll_max_15m_combinedPeak instability over 15 minutes2roll_std_15m_combinedVolatility of instability score3roll_mean_15m_combinedSustained average instability4slope_15m_etco2CO₂ trend — ventilatory trajectory5slope_15m_heart_rateHR trend — cardiac trajectory6pulse_pressureVascular instability marker
All top features represent physiological trajectories, not isolated abnormal values.

Repository Structure
├── catevcode.py                  # Physiological risk engine & feature pipeline
├── catev_cnn_v8_training.py      # Feature engineering & CNN training
└── README.md
Saved model artifacts:
catev_cnn_v8_model.keras          # Trained model weights
catev_cnn_v8_meta.pkl             # Scaler and metadata
catev_cnn_v8_report.txt           # Full evaluation report
catev_cnn_v8_history.png          # Training curves
catev_cnn_v8_validation_cm.png    # Validation confusion matrix
catev_cnn_v8_test_cm.png          # Test confusion matrix

Applications

ICU early warning and real-time deterioration monitoring
Clinical decision support for bedside staff
Multi-organ failure detection research
Physiological instability modeling and dataset construction


Limitations

Single-center dataset — generalizability to other ICU populations is unknown
Critical class classification remains challenging due to its transitional physiological nature
Currently a research prototype, not a certified clinical product
Requires continuous high-frequency vital monitoring at 2-second resolution
Val/Test class distributions differ from training — reflects real patient variability but limits direct comparison


Planned Extensions

 Rule-Based AI Layer — A structured reasoning layer enabling explicit clinical logic to interpret and explain instability classifications without relying solely on learned patterns
 Score Fluctuation Analysis — Studying combined_score trajectories: instability oscillation patterns, transition velocities, and physiological drivers behind score variance
 Validation on larger, multi-hospital datasets
 Transformer / LSTM time-series models
 Real-time ICU deployment pipeline
 Prospective clinical validation study
 Extension to broader ICU age groups


Dataset
The master dataset used for this project is hosted on Kaggle due to GitHub file size limits.
📥 Download the dataset from Kaggle
After downloading, place the dataset file inside the project directory before running the code.

Author
Arjun
MSc Artificial Intelligence & Machine Learning
Indian Institute of Information Technology, Lucknow (IIIT-L)
