# F.T — Flow of Trajectory
### Physiological Instability Detection Engine

> *A time-aware clinical intelligence system for real-time ICU deterioration detection — combining physiological rule modelling, temporal state logic, and deep learning to flag patient instability up to 15 minutes before clinical failure.*

Built on the **VitalDB dataset**: 2,378,857 rows of high-resolution physiological signals from 381 surgical/ICU patients, recorded at 2-second intervals.

---

## Table of Contents

1. [What F.T Does](#1-what-ft-does--three-core-purposes)
2. [Performance at a Glance](#2-performance-at-a-glance)
3. [Why Not NEWS2?](#3-why-not-news2)
4. [System Architecture](#4-full-system-architecture)
5. [Dataset & Input Signals](#5-dataset--input-signals)
6. [Dataset Pipeline](#6-dataset-pipeline--how-the-master-dataset-is-built)
7. [Purpose 1 — Real-Time Severity Classification](#7-purpose-1--real-time-severity-classification)
8. [Purpose 2 — Rule-Based Explainability Layer](#8-purpose-2--rule-based-explainability-layer)
9. [Purpose 3 — 15-Minute Ahead Prediction](#9-purpose-3--15-minute-ahead-deterioration-prediction)
10. [Feature Engineering](#10-feature-engineering)
11. [Repository Structure](#11-repository-structure)
12. [Running the Rule-Based Layer](#12-running-the-rule-based-layer)
13. [Limitations](#13-limitations)
14. [Planned Extensions](#14-planned-extensions)
15. [Quick Reference](#15-quick-reference)

---

## 1. What F.T Does — Three Core Purposes

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

## 2. Performance at a Glance

```
Emergency Detection Rate           ~94%
Binary AUROC (Normal vs At-Risk)   0.7987
Test AUROC                         0.7234
AUPRC                              0.5654
Balanced Accuracy                  0.53
Model Parameters                   275,396
Prediction Horizon                 ~15 minutes ahead
Input Resolution                   2-second vital streams
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

When the model labels a deteriorating patient as **Critical** instead of **Emergency**, that is not an error — it is **conservative early detection**. The alert still fires. The clinician is still called. Both labels trigger intervention; only one is penalised by conventional metrics.

---

## 3. Why Not NEWS2?

Because NEWS2 detects deterioration **after it becomes obvious**. F.T detects it **as it develops**.

| Capability | NEWS2 | F.T |
|---|---|---|
| Approach | Static snapshot scoring | Continuous physiological trajectory |
| Signal source | Current threshold breaches | Evolving multi-vital patterns |
| Pattern detection | Single vital, isolated | Multi-vital combinations across organ systems |
| Temporal consistency | None — single point in time | FSM-stabilised 15-window confirmation |
| Early warning | At threshold breach | Before threshold breach (ramp detection) |
| Hidden instability | Not detected | Tier 3 patterns: Masked Shock, Stable Deceiver, Occult Acidosis |

### Benchmark Results

F.T was benchmarked against a modified NEWS2 implementation adapted for VitalDB's intraoperative constraints. Temperature and consciousness were excluded (not consistently available or meaningful under anaesthesia); RR, SpO₂, HR, SBP, and FiO₂ proxy were used.

| Metric | Agreement |
|---|---|
| Severity Match | 61.39% (33,612 / 54,750 rows) |
| Result Match | 60.86% (33,323 rows) |
| Exact Match (Both) | 54.10% (29,618 rows) |

The ~54% exact match is **expected and informative**. It reflects the fundamental difference in philosophy: NEWS2 asks *"Is this patient currently outside normal bounds?"* — F.T asks *"Is this patient's trajectory heading toward failure?"*

### What NEWS2 Structurally Cannot See

F.T's Tier 3 patterns detect instability that is invisible to any single-vital snapshot system:

- **Stable Deceiver** — SpO₂ 92–94%, HR 75–90, MBP 65–70: vitals that look acceptable while masking early circulatory failure
- **Masked Shock** — MBP 65–72, HR < 90: perfusion decline without the compensatory tachycardia that would normally raise an alarm
- **Occult Acidosis** — ETCO₂ ≤ 32, RR ≥ 24, SpO₂ 88–92: metabolic distress where each individual vital sits just below any single-vital alert threshold

> **F.T does not replace NEWS2. It extends it** — adding early and hidden instability detection, multi-parameter physiological interaction modelling, and time-consistent severity estimation.

---

## 4. Full System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║          Raw ICU Vital Streams  (2-second resolution)            ║
║  SpO₂ · HR · RR · SBP · DBP · MBP · ETCO₂ · Pulse Pressure     ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║                 Data Cleaning & Preprocessing                    ║
║   · Column renaming from Solar8000 monitor codes                 ║
║   · Pulse pressure derivation  (SBP − DBP)                      ║
║   · Per-patient forward-fill → backward-fill imputation          ║
║   · Respiratory rate smoothing  (5-row rolling mean)             ║
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
║   · Early deterioration ramp  (onset 20% before threshold)       ║
║   · Condition amplification multiplier  (capped at 2.2×)        ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║             Temporal Stability Engine  (FSM)                     ║
║   · Label confirmation across 15 consecutive readings            ║
║   · Sensor-noise-driven flickering suppressed                    ║
║   · Emergency → Normal direct transition blocked                 ║
╚══════════════════════════════════════════════════════════════════╝
                                ↓
╔══════════════════════════════════════════════════════════════════╗
║                     Label Construction                           ║
║   · severity_label  — raw score → class                         ║
║   · result_label    — FSM-confirmed class                        ║
║   · future_label    — result_label shifted 450 rows (= 15 min)  ║
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

## 5. Dataset & Input Signals

| Property | Value |
|---|---|
| Source | VitalDB — high-resolution perioperative physiological monitoring |
| Monitor system | Solar8000 bedside monitor (arterial line transducers for BP) |
| Patients | 381 ICU patients, aged 60–80 years |
| Resolution | 2-second intervals |
| Total rows (raw) | ~2,378,857 |

### Input Vital Signals

| Signal | Column | Raw Monitor Code | Notes |
|---|---|---|---|
| SpO₂ | `spo2` | `Solar8000/PLETH_SPO2` | Oxygen saturation |
| Heart Rate | `heart_rate` | `Solar8000/HR` | Pulse rate (bpm) |
| Respiratory Rate | `resp_rate` | `Solar8000/RR` | Smoothed before use; raw column retained |
| Systolic BP | `sbp` | `Solar8000/ART_SBP` | Arterial line transducer |
| Diastolic BP | `dbp` | `Solar8000/ART_DBP` | Arterial line transducer |
| Mean BP | `mbp` | `Solar8000/ART_MBP` | Direct monitor output — **not** derived as (SBP+2×DBP)/3 |
| End-Tidal CO₂ | `etco2` | `Solar8000/ETCO2` | Ventilatory CO₂ marker |
| Pulse Pressure | `pulse_pressure` | Derived: SBP − DBP | First-class feature, not a helper variable |

### Target Label Distribution (`future_label`)

| Class | Label | Share |
|---|---|---|
| 0 | Normal | ~40% |
| 1 | Critical | ~20% |
| 2 | Emergency | ~40% |

---

## 6. Dataset Pipeline — How the Master Dataset Is Built

Complete sequence of operations in `cleaning.ipynb` that transforms raw VitalDB exports into the 99-feature master dataset. Every design decision is documented here.

### Step 1 — Sort by Patient and Time

All rows are sorted by `patient_id` then `time` ascending. This ensures temporal continuity within each patient's signal stream and is a prerequisite for all downstream slope and lag computations.

### Step 2 — Unnamed Column Removal and Time-NaN Dropping

`Unnamed:` index columns from intermediate CSV writes are removed. Rows where `time` itself is `NaN` are dropped — these represent monitor entries with no temporal anchor and cannot be placed in the signal stream.

### Step 3 — Pulse Pressure Derivation

```python
df["pulse_pressure"] = df["sbp"] - df["dbp"]
```

Pulse pressure is not a direct Solar8000 output. It is derived as SBP − DBP and treated as a **first-class physiological feature** throughout the entire system. Its low and wide extremes signal distinct clinical states — stroke volume deficit, vascular stiffness, tamponade — that SBP and DBP individually cannot surface.

### Step 4 — Missing Value Imputation

```python
vital_cols = ["spo2", "heart_rate", "resp_rate", "sbp", "dbp", "mbp", "etco2", "pulse_pressure"]

df[vital_cols] = (
    df.groupby("patient_id")[vital_cols]
      .transform(lambda x: x.ffill().bfill())
)
```

Strategy: **per-patient forward-fill, then backward-fill.** Monitors drop samples during probe reattachment, cable disconnections, or momentary artefacts — the last known reading is the best clinical approximation. Forward-fill covers mid-record gaps; backward-fill handles NaN rows at the start of a patient record. Grouping by `patient_id` ensures no patient's gaps are filled with another patient's values.

### Step 5 — Respiratory Rate Smoothing

```python
RR_SMOOTH_WINDOW = 5   # 5 rows × 2 seconds = 10-second rolling mean
```

Respiratory rate is the noisiest of the eight vitals due to natural inter-breath variability. A 5-row rolling mean produces `resp_rate_smoothed`, used in all downstream scoring, pattern detection, slope, and lag computations. The raw `resp_rate` column is retained.

### Step 6 — Physiological Risk Scoring

For each of the 8 vitals (using `resp_rate_smoothed` for RR), the pipeline computes a continuous abnormality score, transforms it nonlinearly, then aggregates across organs.

**Threshold zones** — three clinical risk zones per vital:

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

**Vital direction map** — not every vital is dangerous in both directions. This is explicitly encoded to prevent spurious alerts:

```python
VITAL_DIRECTION = {
    "spo2": "low",   # SpO₂ = 100% is healthy — only low values are dangerous
    "mbp":  "low",   # MAP threshold is a lower bound only
}
# All others default to "both" — they have explicit _low and _high keys
```

**Continuous abnormality score `z ∈ [0, 1]`** — rather than binary zone membership, each vital maps to a continuous score interpolated between boundaries. The ramp begins 20% before the formal threshold to encode sub-threshold deterioration:

```python
EARLY_FRAC = 0.20
early_start = threshold − 0.20 × (threshold − normal_reference)

z = 0.0  →  Normal
z = 0.5  →  Critical boundary
z = 1.0  →  Emergency boundary
```

For example: if a pattern threshold is MBP < 70 with a normal reference of 90, the ramp begins at MBP = 74. The vital accumulates score before crossing the clinical line.

**Nonlinear severity transformation:**

```
severity = 2^z − 1
```

| z | severity |
|---|---|
| 0.0 | 0.000 |
| 0.5 | 0.414 |
| 1.0 | 1.000 |

Severity accelerates near emergency levels, reflecting the clinically real phenomenon of nonlinear risk escalation. A patient at z = 0.5 is approximately 40% as dangerous as z = 1.0 — not 50%.

**Multi-organ aggregation:**

```
severity_sum = Σ severity_i   (across all 8 vital signs)
```

This simultaneously handles two distinct clinical presentations: a single severely deranged vital (high individual score, moderate sum), and multiple concurrent mild abnormalities across organ systems (low individual scores, elevated sum from accumulation).

### Step 7 — Disease Pattern Detection and Condition Multipliers

The 12 clinical patterns are evaluated per row. Each active pattern computes a continuous **condition multiplier** that amplifies the `combined_score` — going beyond a binary flag.

A component vital must exceed `MIN_COND_ACTIVATION = 0.20` on its z-score to contribute to the pattern factor. The same 20% early ramp applies, so patterns begin partially activating before their formal trigger is fully met.

**Tier-based multipliers:**

```python
BASE_1_2 = 1.4672   # Tier 1 & Tier 2 base
BASE_3   = 1.19     # Tier 3 base — lower because Tier 3 is subtle/hidden instability

EXTRA_T1 = 1.30     # Additional factor when a Tier 1 pattern is active
EXTRA_T2 = 1.20     # Additional factor when a Tier 2 pattern is active
EXTRA_T3 = 1.10     # Additional factor when a Tier 3 pattern is active

MULTIPLIER_CAP = 2.2   # Hard cap — prevents runaway escalation when multiple patterns co-activate

combined_score = severity_sum × min(condition_multiplier, MULTIPLIER_CAP)


```
why this number-
The base curve 
2^x−1 reaches 1 at x = 1
With 1.4672 scaling:
1.4672(2^0.75-1)~1

✔️ Meaning:
The same critical severity level is reached at x ≈ 0.75 instead of 1
This creates ~25% earlier escalation
Models rapid physiological deterioration in Tier 1 & 2 conditions

👉 Interpretation:
High-risk conditions are made to escalate faster than natural progression
Mild scaling compared to 1.4672
Reaches severity = 1 around
x~0.8 to 0.9
✔️ Meaning:
Only slight acceleration (~19%)
Keeps progression close to natural curve
Captures subtle or hidden instability without over-triggering

👉 Interpretation:
Tier 3 conditions progress, but not aggressively

. Early ramp (0.20)

Physiological deterioration is gradual, not binary. Clinical states begin to manifest before formal thresholds are crossed.

A 20% early ramp allows partial activation when a vital is approaching its pathological limit.
It ensures:
Smooth transition from 0 → 1 activation
Capturing pre-threshold instability
Justification:
< 0.1 → too insensitive (misses early warning)
> 0.3 → too aggressive (causes premature triggering)
0.2 provides a balanced onset of activation

Additionally, the ramp only contributes when other vitals already meet condition criteria, preserving clinical validity.
Extra multipliers (1.3, 1.2, 1.1)

These model multi-condition interaction severity:

Tier 1 → +30% amplification
Tier 2 → +20% amplification
Tier 3 → +10% amplification

This reflects:

Severe conditions dominate risk amplification
Mild patterns contribute incrementally

Only the top two impactful patterns are used to:

Avoid over-counting noise
Maintain clinical interpretability

 Multiplier cap (2.2)-

Prevents runaway escalation when multiple patterns co-activate.

Ensures system remains:
Stable
Clinically realistic
Avoids false emergency inflation



Tier 3 carries a lower base (1.19 vs 1.4672) because Tier 3 patterns represent clinically significant but not acutely life-threatening instability. Over-amplifying subtle presentations would produce false emergencies for what are still early-stage warning signals.

**The 12 patterns:**

*Tier 1 — Major Instability*

| Pattern | Vitals | Thresholds | Normal Refs | Clinical Meaning |
|---|---|---|---|---|
| Shock Spiral | MBP ↓, HR ↑ | MBP < 70, HR > 100 | 90, 75 | Low perfusion with compensatory tachycardia |
| Respiratory Burnout | SpO₂ ↓, RR ↑ | SpO₂ < 92, RR > 22 | 98, 16 | Oxygen failure with increased respiratory effort |
| Hypercapnic Failure | ETCO₂ ↑, RR ↓ | ETCO₂ > 50, RR < 10 | 40, 16 | Ventilatory failure with CO₂ retention |

*Tier 2 — Moderate Risk*

| Pattern | Vitals | Thresholds | Normal Refs |
|---|---|---|---|
| Pulse Pressure Low | PP ↓ | PP < 30 | 50 |
| Wide PP + High SBP | PP ↑, SBP ↑ | PP > 70, SBP > 170 | 50, 120 |
| Respiratory-Haemodynamic Combo | SpO₂ ↓, RR ↑, HR ↑ | SpO₂ < 92, RR > 22, HR > 100 | 98, 16, 75 |

*Tier 3 — Subtle / Hidden Risk (the patterns NEWS2 cannot see)*

| Pattern | Vitals | Thresholds | Normal Refs | Why Dangerous |
|---|---|---|---|---|
| Hypertensive Emergency | SBP ↑, PP ↑ | SBP > 180, PP > 70 | 120, 50 | Extreme pressure with wide pulse |
| Stable Deceiver | SpO₂ ↓, HR ↓, MBP ↓ | SpO₂ < 94, HR < 90, MBP < 70 | 98, 100, 90 | Acceptable-looking vitals masking early circulatory failure |
| Masked Shock | MBP ↓, HR ↓ | MBP < 72, HR < 90 | 90, 100 | Perfusion decline without compensatory tachycardia |
| Occult Acidosis | ETCO₂ ↓, RR ↑, SpO₂ ↓ | ETCO₂ < 32, RR > 24, SpO₂ < 92 | 40, 16, 98 | Metabolic distress below all overt alarm values |
| Trend Decline | ETCO₂ ↑, SpO₂ ↓, HR ↑ | ETCO₂ > 50, SpO₂ < 92, HR > 100 | 40, 98, 75 | Coordinated multi-vital drift toward failure |
| Trend Activate | SpO₂ ↓, HR ↑, ETCO₂ ↓ | SpO₂ < 94, HR > 95, ETCO₂ < 35 | 98, 75, 40 | Slope-based trajectory signal before threshold breach |

### Step 8 — OLS Slope Features (32 + 4)

For each of the 8 raw vitals, OLS slopes are computed at four lookback horizons:

```python
SLOPE_WINDOWS = {"2m": 60, "5m": 150, "7m": 210, "15m": 450}
# rows — each row = 2 seconds
```

OLS is preferred over point-to-point difference because a single noisy reading does not dominate the estimate. The slope captures the **sustained trend direction** across the window. This produces 32 vital slope features plus 4 combined-score slopes: `slope_{2m,5m,7m,15m}_combined_score`.

### Step 9 — Rolling Statistics (10 Features)

Rolling mean, standard deviation, min (15m), and max (15m) are computed on `combined_score` across the four time windows. These capture score stability over time: high mean + low standard deviation = sustained danger; high standard deviation = oscillating instability — a distinct clinical presentation that single-point scoring misses entirely.

### Step 10 — Lag Features (9 Features)

```python
LAG_15M = 450   # 900 seconds = 15 minutes at 2-second resolution
```

A 15-minute lookback value is created for each of the 8 raw vitals plus `combined_score` by shifting 450 rows backward within each patient group. These serve three roles: (1) the Rule-Based Layer uses them to compare current state to 15 minutes ago without loading extra rows; (2) Stage 5 recovery detection compares lag abnormality against current slope direction; (3) Stage 4 trend warnings only fire for vitals that were already abnormal at the lag point, preventing false alarms on vitals starting from a normal baseline.

### Step 11 — NaN Cleanup

Remaining NaNs from rolling and lag window edge effects are filled with 0. Intermediate helper columns (`selected_cond_1`, `selected_cond_2`) are dropped.

### Step 12 — Label Assignment

**Severity label** — raw instantaneous classification from `combined_score`:

```python
CRITICAL_THRESHOLD  = 0.75
EMERGENCY_THRESHOLD = 1.4

severity_label:
  0  →  Normal      (combined_score < 0.75)
  1  →  Critical    (0.75 ≤ combined_score < 1.4)
  2  →  Emergency   (combined_score ≥ 1.4)
```

**Result label** — FSM-confirmed label (see Section 7 for full FSM logic).

**Future label** — the prediction target: `result_label` shifted 450 rows (15 minutes) forward within each patient group. Derived from `result_label` — not `severity_label` — so the target represents a confirmed, stable physiological state rather than a noisy instantaneous reading.

### Step 13 — Edge Row Trimming

Two trimming operations ensure every row has fully populated features:

```python
# Tail trim — remove last 450 rows per patient (no future_label beyond this point)
df = df.groupby('patient_id', group_keys=False).apply(
    lambda x: x.iloc[:-450] if len(x) > 450 else pd.DataFrame()
)

# Head trim — remove first 450 rows per patient (lag + slope windows not yet populated)
df = df.groupby("patient_id").apply(lambda x: x.iloc[450:]).reset_index(drop=True)
```

After both trims, every row has a valid `future_label`, a complete 15-minute lag value, and fully populated slope features across all four horizons.

---

## 7. Purpose 1 — Real-Time Severity Classification

The scoring pipeline from Section 6 (Steps 6–7) produces `combined_score` in real time. The FSM below converts that score into a stable, clinically actionable severity label.

### Temporal Stability Engine — Hierarchical FSM

A Finite State Machine prevents label flickering caused by sensor noise. It operates over a sliding window of 15 consecutive severity readings (30 seconds at 2-second resolution).

**Why 15 readings?** Long enough to reject probe artefacts (typically 1–3 readings); short enough to confirm genuine deterioration (which persists over minutes). 30 seconds of sustained signal before confirmation is clinically appropriate.

**FSM parameters:**

```python
WINDOW_SIZE             = 15   # Sliding observation window
CONFIRM_LEN             = 15   # Hard consecutive count — immediately locks in new state
EMERGENCY_UPGRADE_COUNT = 10   # ≥10 Emergency in window → upgrade from Critical
NORMAL_DOWNGRADE_COUNT  = 12   # ≥12 Normal in window → downgrade from Critical
```

**Transition rules:**

| Transition | Condition |
|---|---|
| Any → Any (hard confirm) | 15 consecutive identical readings immediately lock in the label |
| Emergency downgrade | Window contains zero Emergency readings → step down to Critical |
| Critical → Emergency | Window contains ≥10 Emergency (no Critical) → upgrade |
| Critical → Normal | Window contains ≥12 Normal (no Critical) → downgrade |
| Normal → Critical | Window contains zero Normal readings → step up |
| **Emergency → Normal** | **Blocked** — must pass through Critical first |

**Why Emergency cannot go directly to Normal:** A patient recovering from an emergency state does not instantly return to normal physiology. Organs that experienced ischaemia or haemodynamic failure continue to show abnormal compensatory signals during recovery. Requiring passage through Critical prevents premature declaration of stability while the patient is still recovering.

---

## 8. Purpose 2 — Rule-Based Explainability Layer

Modern clinical AI models often produce risk scores without explaining their reasoning — a fundamental barrier to clinical adoption. The rule-based layer wraps the severity classification in a fully transparent, deterministic reasoning pipeline. It produces a structured clinical report for any patient at any time point.

Every output traces back to a specific vital, a specific threshold, and a specific clinical rationale. No black-box inference. Dependencies: `numpy`, `pandas` only.

### 5-Stage Pipeline

```
Patient Snapshot (patient_id + time)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1 — System Label Display                                  │
│  severity_label vs result_label shown side by side               │
│  with the combined_score value                                   │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2 — FSM Mismatch Explanation                              │
│  Four distinct cases with plain-language outputs:                │
│  (a) Both agree — Normal → patient stable, confirmed             │
│  (b) Both agree — Critical/Emergency → sustained deterioration   │
│  (c) Current = Normal, FSM still elevated                        │
│      → watchful holding state; FSM awaiting sustained recovery   │
│  (d) Current elevated, FSM at different level                    │
│      → state transition in progress; FSM label explained         │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3 — Vital Sign Analysis                                   │
│  Early exit with watch note if current severity = Normal         │
│                                                                  │
│  Step A — Per-vital triggers (direction-aware lookup)            │
│    Each triggered vital: value, severity level, threshold,       │
│    and a plain-language clinical note                            │
│                                                                  │
│  Step B — Active pattern flags                                   │
│    All 12 conditions checked; active ones listed with tier       │
│                                                                  │
│  Step C — All out-of-range vitals                                │
│    Comprehensive list including Critical and Borderline levels   │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 4 — Trend & Noise Warnings                                │
│                                                                  │
│  Warning 1 — Monotonic slope acceleration                        │
│    Only fires for vitals that were abnormal at the lag point     │
│    Accelerating worsening: slope_15m < slope_7m < slope_5m < slope_2m  │
│    Reports direction, rate of change, and current lag value      │
│                                                                  │
│  Warning 2 — Physiologically impossible values                   │
│    PP = 0 mmHg · SpO₂ < 20% · HR > 250 bpm                      │
│    RR > 60 /min · SBP < 10 mmHg                                  │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 5 — Recovery Detection                                    │
│  For each vital — was it abnormal at the 15-min lag?             │
│  · HIGH-abnormal 15m ago + all 4 slope windows now negative      │
│    → vital returning toward normal (good sign)                   │
│  · LOW-abnormal 15m ago + all 4 slope windows now positive       │
│    → vital returning toward normal (good sign)                   │
│  Reports: lag value, current value, all four slope values        │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
╔══════════════════════════════════════════════════════════════════╗
║  Final Report                                                    ║
║  Raw Severity · FSM Confirmed · 15-Min Forecast label            ║
║  Active condition flags with tier · Trend & recovery summaries  ║
║  Status: STABLE / WATCH STATE / TRANSITIONING /                  ║
║  CONFIRMED CRITICAL / CONFIRMED EMERGENCY                        ║
╚══════════════════════════════════════════════════════════════════╝
```

### Stage 4 — Design Decisions

**Why four nested windows for trend detection?** A single slope (15m → now) is misleading if the trend reversed halfway. Checking four nested windows (15m → 7m → 5m → 2m) confirms the worsening is continuous and accelerating — not a temporary spike. Only vitals abnormal at the lag point are monitored, preventing false alarms on vitals that started from a normal baseline.

**Physiologically impossible values** flag sensor or equipment failures before they corrupt the classification:

| Vital | Impossible Threshold | Clinical Reason |
|---|---|---|
| Pulse Pressure | PP = 0 mmHg | Physiologically impossible in a living patient |
| SpO₂ | SpO₂ < 20% | Incompatible with life — almost certainly a probe artefact |
| Heart Rate | HR > 250 bpm | Exceeds physiological limit of the conduction system |
| Respiratory Rate | RR > 60 /min | Mechanically impossible without ventilator malfunction |
| Systolic BP | SBP < 10 mmHg | Incompatible with cerebral and cardiac perfusion |

### Stage 5 — Recovery Confirmation Logic

All four slope windows must agree before a recovery is reported. If the 5-minute slope is improving but the 2-minute slope has turned negative, the recovery is not confirmed — it may be a brief fluctuation before further decline. Each confirmed recovery vital shows: its lag value (where it was 15m ago), current value, and all four slope values to make the trend explicit.

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

  🚨 Both labels agree: EMERGENCY confirmed.
     Deterioration sustained across the 15-reading window.
     The FSM has locked in this label.
     Immediate clinical attention is required.
═════════════════════════════════════════════════════════════════
  STAGE 3 — VITAL SIGN ANALYSIS

  STEP A — EMERGENCY-THRESHOLD VITAL TRIGGERS

  👉 Emergency triggered by: [Respiratory Rate, Systolic Blood Pressure,
     Mean Arterial Pressure, Pulse Pressure]

  🔴 Respiratory Rate
       Current Value  : 40.0 breaths/min
       Status         : Emergency — above expected range of 20 /min
       Clinical Note  : Patient is breathing faster than normal.
                        May be compensating for low oxygen or acidosis.

  🔴 Systolic Blood Pressure
       Current Value  : 89.0 mmHg
       Status         : Emergency — below expected range of 110 mmHg
       Clinical Note  : Hypotension — heart may not be pumping
                        sufficient blood to vital organs.

  🔴 Mean Arterial Pressure
       Current Value  : 62.0 mmHg
       Status         : Critical — below expected range of 70 mmHg
       Clinical Note  : MAP below 70 mmHg associated with
                        organ ischaemia and shock.

  🔴 Pulse Pressure
       Current Value  : 43.0 mmHg
       Status         : Borderline-Low
       Clinical Note  : Warning sign of tamponade,
                        severe hypovolaemia, or cardiogenic shock.
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

## 9. Purpose 3 — 15-Minute Ahead Deterioration Prediction

### How Early Warning Is Achieved

Standard threshold-based systems trigger when a vital crosses a boundary. F.T begins accumulating signal **before** any boundary is reached — via the 20% early deterioration ramp introduced in Step 6. The slope features, rolling statistics, and combined score trajectories fed to the CNN-GRU carry this sub-threshold signal forward. The model predicts deterioration based on **trajectory**, not position.

### CNN-GRU v7 Architecture

```
Input: (batch, 80 timesteps × 44 features)
          ↓
  Multi-scale Conv1D
  (kernel sizes 7 → 5 → 3, channels 48 → 72 → 72)
  + Residual connections
          ↓
  2-layer Bidirectional GRU  (hidden = 48, output = 96)
          ↓
  Attention Pooling
          ↓
  Temperature Scaling  (T = 1.49)
          ↓
  Output: Normal / Critical / Emergency  (15 min ahead)

  Total parameters: 275,396
```

**Why CNN + GRU?** The multi-scale Conv1D layers extract local physiological patterns at different time scales simultaneously — short transients and sustained trends in the same pass. The bidirectional GRU then reads these patterns sequentially, capturing how the physiological state evolves over the 80-step (~2.7-minute) input window. Attention pooling weights which timesteps matter most. Residual connections preserve gradient flow through the convolutional stack.

**Why temperature scaling (T = 1.49)?** Without calibration, the model's softmax probabilities are overconfident — inflated toward extreme values. Post-hoc temperature scaling produces well-calibrated probability estimates. In a clinical setting, the *degree* of certainty is as important as the predicted class.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Batch size | 256 |
| Max epochs | 60 (early stopped at epoch 29) |
| Loss function | Focal Loss |
| Regularization | SWA + EMA smoothing + Jitter augmentation |
| Post-hoc calibration | Temperature scaling (T = 1.49) |

**Why Focal Loss?** Class distribution is imbalanced (Normal ~40%, Critical ~20%, Emergency ~40%). Focal Loss down-weights well-classified examples, forcing the model to focus on hard cases — which clinically are the Critical/Emergency boundary cases that matter most.

### Dataset Split — Patient-Level

Splits are patient-level, not row-level, to prevent data leakage. A patient appearing in both train and test would allow the model to memorise patient-specific physiological patterns rather than generalise.

| Split | Windows | Normal | Critical | Emergency |
|---|---|---|---|---|
| Train | 57,300 | 23,028 | 11,660 | 22,612 |
| Val | 6,630 | 2,323 | 1,583 | 2,724 |
| Test | 6,876 | 2,109 | 1,547 | 3,220 |

### Full Performance Metrics

| Metric | Value |
|---|---|
| Emergency Detection Rate (Critical + Emergency combined) | ~94% |
| Binary AUROC (Normal vs at-risk) | 0.7987 |
| Test AUROC | 0.7234 |
| AUPRC | 0.5654 |
| Balanced Accuracy | 0.53 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal (0) | 0.69 | 0.42 | 0.52 | 2,109 |
| Critical (1) | 0.30 | 0.64 | 0.41 | 1,547 |
| Emergency (2) | 0.75 | 0.55 | 0.63 | 3,220 |

**Confusion matrix:**

| | Pred Normal | Pred Critical | Pred Emergency |
|---|---|---|---|
| True Normal | 890 | 978 | 241 |
| True Critical | 215 | 985 | 347 |
| True Emergency | 71 | 1,646 | 1,757 |

Critical class precision (0.30) reflects inherent label boundary ambiguity — Critical and Emergency overlap physiologically, and ground truth labels carry uncertainty at this boundary. The system is intentionally tuned toward recall: a false alarm is far less costly than a missed deterioration.

---

## 10. Feature Engineering

F.T separates two concerns: the **master dataset** (full analytical coverage, 99 features — used by the Rule-Based Layer) and the **model input set** (curated interpretable subset — used by CNN-GRU).

**Design principle:** Every model input maps directly to a physiological concept. PCA, latent embeddings, and black-box statistical constructs are deliberately excluded. The system can explain every feature it acts on.

### Master Dataset — 99 Features

| Category | Count | Features |
|---|---|---|
| Identifiers | 2 | patient_id · time |
| Raw Vitals | 9 | dbp · mbp · heart_rate · resp_rate · sbp · spo2 · etco2 · pulse_pressure · resp_rate_smoothed |
| Vital Slopes (2m, 5m, 7m, 15m) | 32 | OLS slope for each of 8 vitals × 4 horizons |
| Continuous Abnormality Scores | 8 | z_spo2 · z_hr · z_rr · z_sbp · z_dbp · z_mbp · z_etco2 · z_pp |
| Scaled Severity Scores | 8 | s_spo2 · s_hr · s_rr · s_sbp · s_dbp · s_mbp · s_etco2 · s_pp |
| Physiological Instability Scores | 2 | severity_sum · combined_score |
| Disease Pattern Flags | 12 | t1_shock_spiral · t1_resp_burnout · t1_hypercapnic · t2_pulse_pressure_low · t2_widepp_highsbp · t2_resp_hemo_combo · t3_hyper_emergency · t3_stable_deceiver · t3_masked_shock · t3_occult_acidosis · t3_trend_decline · t3_trend_activate |
| Combined Score Slopes (2m, 5m, 7m, 15m) | 4 | slope_{2m/5m/7m/15m}_combined_score |
| Rolling Statistics | 10 | roll_mean/std across multiple windows + roll_min/max_15m_combined |
| Lag Features (15m lookback) | 9 | lag_15m for all 8 vitals + lag_15m_combined_score |
| Labels | 3 | severity_label · result_label · future_label |

The z-scores, pattern flags, 2m vital slopes, shorter-window rolling stats, and lag features are retained in the master dataset for the Rule-Based Layer but excluded from CNN-GRU inputs. The model uses a curated subset; the explainability layer uses everything.

---

## 11. Repository Structure

```
├── cleaning.ipynb          # Full pipeline: cleaning → feature engineering → labelling
├── RULE_BASED_AI.ipynb     # Rule-Based Clinical Reasoning Layer (all 5 stages)
├── cnn_gru_2.py            # Feature engineering & CNN-GRU v7 training
├── README.md
├── news-2/                 # NEWS2 labelling and comparison analysis
└── tier_combination/       # Tier pattern explanation and documentation
```

---

## 12. Running the Rule-Based Layer

```python
import pandas as pd

# Configuration
TARGET_PATIENT_ID = 64
TARGET_TIME       = 3600   # seconds (3600 = 60 min into monitoring)

# Load data
df = pd.read_csv("MASTERDATA.csv")
print(f"Loaded: {df.shape[0]:,} rows | {df['patient_id'].nunique()} patients")

# Extract current and 15-min-ago snapshots
current_stat = df[
    (df['patient_id'] == TARGET_PATIENT_ID) &
    (df['time'] == TARGET_TIME)
].reset_index(drop=True)

lag_stat = df[
    (df['patient_id'] == TARGET_PATIENT_ID) &
    (df['time'] == TARGET_TIME - 900)
].reset_index(drop=True)

if current_stat.empty:
    raise ValueError(f"No data for Patient {TARGET_PATIENT_ID} at time {TARGET_TIME}s.")

past_stat = None if lag_stat.empty else lag_stat

# Run all stages
curr_state, conf_state = stage_1_2(current_stat)
stage_3(current_stat, curr_state, conf_state)
stage_4(current_stat)
stage_5(current_stat, past_stat)
final_report(current_stat, past_stat)
```

**Dataset download** — hosted on Kaggle due to GitHub file size limits:

| Dataset | Description | Link |
|---|---|---|
| Initial Dataset | Raw data before cleaning | [Download from Kaggle](https://www.kaggle.com) |
| Master Dataset | Processed, 99-feature dataset | [Download from Kaggle](https://www.kaggle.com) |

---

## 13. Limitations

| Limitation | Detail |
|---|---|
| Low Critical precision | Inherent label boundary ambiguity — Critical and Emergency overlap physiologically. System tuned toward recall over precision. |
| Single-centre data | Trained on VitalDB only. Generalisability to other ICU populations or hospital settings is unknown. |
| No clinical validation | Research prototype. Not prospectively validated. Not a certified medical device. |
| Intraoperative context | VitalDB captures surgical/perioperative monitoring — dynamics may differ from a general ICU population. |
| Hardware dependency | Requires continuous high-frequency monitoring at 2-second resolution, not available in all clinical settings. |
| NEWS2 comparison scope | Modified NEWS2 excluded temperature and consciousness due to data constraints; full comparison pending. |
| Age range | Trained on patients aged 60–80 years. Performance outside this range has not been evaluated. |

---

## 14. Planned Extensions

- **XAI Integration** — SHAP / attention visualisation to complement rule-based explanations with deep learning feature attribution
- **Score Fluctuation Analysis** — Instability oscillation patterns, transition velocities, and physiological drivers behind score variance
- **Transformer architecture** — Attention-based architecture exploration for extended datasets
- **LSTM comparison study** — Benchmarking against the CNN-GRU architecture
- **Real-time ICU deployment pipeline**
- **Prospective clinical validation study**
- **Multi-hospital validation** — Generalisability across different ICU populations
- **Extended age range** — Validation below 60 and above 80 years
- **Full NEWS2 comparison** — Including temperature and consciousness with a suitable dataset

---

## 15. Quick Reference

| Property | Value |
|---|---|
| Target population | ICU patients aged 60–80 |
| Input vitals | SpO₂, HR, RR, SBP, DBP, MBP, ETCO₂, Pulse Pressure |
| Data resolution | 2-second intervals |
| Monitor source | VitalDB / Solar8000 |
| Prediction window | ~2.7-min input → 15 min early warning |
| Severity classes | Normal / Critical / Emergency |
| Classification thresholds | < 0.75 Normal · 0.75–1.4 Critical · ≥ 1.4 Emergency |
| Master dataset features | 99 |
| Model architecture | CNN-GRU v7 (275,396 parameters) |
| Emergency detection rate | ~94% |
| Binary AUROC (Normal vs at-risk) | 0.7987 |
| Test AUROC | 0.7234 |
| FSM confirmation window | 15 consecutive readings (30 seconds) |
| Condition multiplier cap | 2.2× |
| Early ramp onset | 20% before formal threshold |
| Tier multipliers | T1/T2 base: 1.4672 · T3 base: 1.19 |
| Extra tier factors | T1: ×1.30 · T2: ×1.20 · T3: ×1.10 |

---

> **F.T is a research prototype. It is not intended for direct clinical use without prospective validation and regulatory clearance.**
