# ============================================================
#  Patient Vitals Early Warning System
#  Built for actual dataset structure (confirmed features)
#
#  DATASET COLUMNS (confirmed):
#    Identifiers : patient_id, time
#    Raw vitals  : spo2, heart_rate, resp_rate, sbp, dbp,
#                  mbp, etco2, pulse_pressure, resp_rate_smoothed
#    Z-scores    : z_spo2, z_hr, z_rr, z_sbp, z_dbp, z_mbp, z_etco2, z_pp
#    S-scores    : s_spo2, s_hr, s_rr, s_sbp, s_dbp, s_mbp, s_etco2, s_pp
#    Scores      : severity_sum, combined_score
#    Flags       : any_vital_E, any_vital_C, tier1_any, tier2_any,
#                  t3_hyper_emergency, t3_stable_deceiver, t3_masked_shock,
#                  t3_occult_acidosis, t3_trend_decline
#    Existing    : s_*_slope_2m, s_*_slope_5m, s_*_slope_7m
#                  severity_sum_slope_2m/5m/7m, combined_score_slope_2m/5m/7m
#    FSM labels  : severity_label, result_label  (EXCLUDED from features)
#    Target      : future_label (0=Normal, 1=Critical, 2=Emergency)
#
#  ADDED HERE (longer memory for 20-min prediction):
#    s_*_slope_15m, severity_sum_slope_15m, combined_score_slope_15m
#    accel_* (slope of 7m-slope — is deterioration speeding up?)
#    cross_score_momentum (15m vs 7m slope divergence)
#
#  TRAINING APPROACH:
#    - Stratified patient split (sort by Emergency% → interleave)
#    - Episode sampling on train only (break FSM sticky runs)
#    - Val/test keep ALL rows (honest real-world evaluation)
# ============================================================

import warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap, joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score,
                              ConfusionMatrixDisplay)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED)

CLASS_NAMES   = ['Normal', 'Critical', 'Emergency']
TARGET_COL    = 'future_label'

# Boost Critical heavily — it's the most dangerous and hardest class
CLASS_WEIGHTS = {0: 1.0, 1: 5.0, 2: 1.0}

STEPS_PER_MIN = 30    # 2s timestep → 30 rows = 1 minute

# Scales we will ADD (2m/5m/7m already exist in dataset)
NEW_SCALE_STEPS = {
    '15m': 15 * STEPS_PER_MIN,   # 450 rows
}

# Columns to extend with 15m/30m slopes (s-scores + aggregate scores)
SLOPE_TARGET_COLS = [
    's_spo2','s_hr','s_rr','s_sbp','s_dbp','s_mbp','s_etco2','s_pp',
    'severity_sum','combined_score'
]

print("=" * 65)
print("Patient Vitals Early Warning — Final Model")
print(f"Target: {TARGET_COL} | Classes: {CLASS_NAMES}")
print("=" * 65)


# ════════════════════════════════════════════════════════════
# EXPANDING SLOPE  — no NaN, no warmup rows lost
#
# At row i: uses min(i, max_W) rows of history.
# Row 0 = 0.0 (no history). Row max_W+ = full window.
# ════════════════════════════════════════════════════════════
def expanding_slope(series, max_W, timestep=2):
    vals   = series.values.astype(np.float64)
    n      = len(vals)
    result = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        W         = min(i, max_W)
        result[i] = (vals[i] - vals[i - W]) / (W * timestep)
    return pd.Series(result, index=series.index)


# ════════════════════════════════════════════════════════════
# 1.  LOAD
# ════════════════════════════════════════════════════════════
def load_data(filepath):
    sep = '\t' if filepath.endswith('.tsv') else ','
    df  = pd.read_csv(filepath, sep=sep)
    df  = df.sort_values(['patient_id', 'time']).reset_index(drop=True)

    # Encode string labels → int if needed
    for col in [TARGET_COL, 'result_label', 'severity_label']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'Normal': 0, 'Critical': 1, 'Emergency': 2})

    # Drop rows where target is NaN
    # (your dataset script removes last N rows per patient)
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    n_patients = df['patient_id'].nunique()
    print(f"\nLoaded  : {len(df):,} rows | {n_patients} patients")
    if len(df) < before:
        print(f"Dropped : {before - len(df):,} NaN target rows")

    print(f"\nTarget distribution ({TARGET_COL}):")
    vc = df[TARGET_COL].value_counts().sort_index()
    for cls, cnt in vc.items():
        print(f"  {CLASS_NAMES[int(cls)]:9s} ({int(cls)}): "
              f"{cnt:7,}  ({cnt / len(df) * 100:.1f}%)")

    rpp = df.groupby('patient_id').size()
    print(f"\nRows/patient: min={rpp.min()}  "
          f"mean={rpp.mean():.0f}  max={rpp.max()}")

    # Per-patient distribution — important for stratified split
    print(f"\nPer-patient class distribution:")
    print(f"  {'Patient':>8}  {'Rows':>6}  {'Normal%':>8}  "
          f"{'Critical%':>10}  {'Emergency%':>11}")
    for pid, grp in df.groupby('patient_id'):
        n_rows = len(grp)
        no = (grp[TARGET_COL] == 0).mean() * 100
        cr = (grp[TARGET_COL] == 1).mean() * 100
        em = (grp[TARGET_COL] == 2).mean() * 100
        print(f"  {pid:>8}  {n_rows:>6,}  {no:>7.1f}%  "
              f"{cr:>9.1f}%  {em:>10.1f}%")

    return df


# ════════════════════════════════════════════════════════════
# 2.  ADD 15m MEMORY + ACCELERATION
#
#  Dataset already has: 2m / 5m / 7m slopes, cross-vital signals.
#  Only adding what is genuinely missing for 20-min prediction:
#    15m slope  — medium-term trajectory (gap between 7m and prediction)
#    accel      — slope of the 7m-slope (is deterioration speeding up?)
#    cross_score_momentum — 15m vs 7m slope divergence on combined_score
# ════════════════════════════════════════════════════════════
def add_extended_memory(df):
    print("\nAdding extended memory features (15m slopes + acceleration)...")
    g   = df.groupby('patient_id', sort=False)
    W15 = NEW_SCALE_STEPS['15m']

    # ── A. 15m slopes on s-scores and aggregate scores ───────
    print("  A. Slopes 15m (expanding — no NaN)...")
    for col in SLOPE_TARGET_COLS:
        if col not in df.columns:
            continue
        new_col = f"{col}_slope_15m"
        if new_col not in df.columns:
            df[new_col] = g[col].transform(
                lambda x, w=W15: expanding_slope(x, w))

    # ── B. Acceleration — slope of the 7m-slope over 15m ─────
    # Answers: is deterioration SPEEDING UP or SLOWING DOWN?
    # positive accel = rate of worsening is increasing
    print("  B. Acceleration (slope of 7m-slope over 15m window)...")
    for col in SLOPE_TARGET_COLS:
        s7_col  = f"{col}_slope_7m"
        acc_col = f"{col}_accel"
        if s7_col not in df.columns:
            continue
        if acc_col not in df.columns:
            df[acc_col] = g[s7_col].transform(
                lambda x, w=W15: expanding_slope(x, w))

    # ── C. Score momentum — 15m vs 7m slope divergence ───────
    # positive = 15m trend worse than 7m = sustained long worsening
    s15 = 'combined_score_slope_15m'
    s7  = 'combined_score_slope_7m'
    if s15 in df.columns and s7 in df.columns:
        df['cross_score_momentum'] = df[s15] - df[s7]
        print("  C. cross_score_momentum (combined_score slope 15m − 7m)...")

    # Verify no NaN
    new_cols = [c for c in df.columns if any(s in c for s in
                ['slope_15m', 'accel', 'cross_score_momentum'])]
    nan_n = df[new_cols].isna().sum().sum()
    print(f"\n  Added  : {len(new_cols)} features")
    print(f"  NaN    : {nan_n} {'✓ (none)' if nan_n == 0 else '⚠ unexpected NaN'}")
    return df


# ════════════════════════════════════════════════════════════
# 3.  STRATIFIED PATIENT SPLIT
#
#  With only 31 patients, random split causes train/val to have
#  very different Emergency%. Stratified split fixes this.
#
#  Method: sort patients by Emergency%, deal them round-robin
#  across splits — like dealing cards.
# ════════════════════════════════════════════════════════════
def split_patients_stratified(df, val_frac=0.15, test_frac=0.15):
    # Per-patient Emergency%
    stats = (df.groupby('patient_id')[TARGET_COL]
               .apply(lambda x: (x == 2).mean())
               .reset_index()
               .rename(columns={TARGET_COL: 'em_pct'})
               .sort_values('em_pct')
               .reset_index(drop=True))

    n       = len(stats)
    n_val   = max(2, round(n * val_frac))
    n_test  = max(2, round(n * test_frac))

    # Deal round-robin: every 3rd to val, every 3rd+1 to test
    train_pids, val_pids, test_pids = set(), set(), set()
    v_count = t_count = 0

    for i, row in stats.iterrows():
        pid = row['patient_id']
        if v_count < n_val and i % 3 == 1:
            val_pids.add(pid); v_count += 1
        elif t_count < n_test and i % 3 == 2:
            test_pids.add(pid); t_count += 1
        else:
            train_pids.add(pid)

    # Any patient not yet assigned → train
    all_pids    = set(df['patient_id'].unique())
    unassigned  = all_pids - train_pids - val_pids - test_pids
    train_pids |= unassigned

    def dist(pids, name):
        sub = df[df['patient_id'].isin(pids)]
        no  = (sub[TARGET_COL] == 0).mean() * 100
        cr  = (sub[TARGET_COL] == 1).mean() * 100
        em  = (sub[TARGET_COL] == 2).mean() * 100
        print(f"  {name:6}: {len(pids):2d} patients | {len(sub):7,} rows | "
              f"N={no:.1f}%  Cr={cr:.1f}%  Em={em:.1f}%")

    print(f"\nStratified patient split:")
    dist(train_pids, 'Train')
    dist(val_pids,   'Val')
    dist(test_pids,  'Test')
    return train_pids, val_pids, test_pids


# ════════════════════════════════════════════════════════════
# 4.  EPISODE SAMPLING — TRAIN ONLY
#
#  FSM creates runs of hundreds of identical-label rows.
#  Model trained on all of them memorizes per-patient baselines.
#  Fix: keep only a few rows per continuous same-label run.
#
#  Applied to TRAIN only. Val/test keep ALL rows.
# ════════════════════════════════════════════════════════════
def episode_sample(df, train_pids, rows_per_episode=5):
    keep = []
    for pid in train_pids:
        grp    = df[df['patient_id'] == pid].reset_index(drop=False)
        labels = grp[TARGET_COL].values
        orig   = grp['index'].values

        # Find where label changes
        changes    = np.where(np.diff(labels) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes, [len(labels)]])

        for s, e in zip(boundaries[:-1], boundaries[1:]):
            ep = orig[s:e]
            if len(ep) <= rows_per_episode:
                keep.extend(ep.tolist())
            else:
                # Always keep first and last
                selected = [ep[0], ep[-1]]
                inner    = ep[1:-1]
                n_inner  = rows_per_episode - 2
                if n_inner > 0 and len(inner) > 0:
                    step = max(1, len(inner) // n_inner)
                    selected.extend(inner[::step][:n_inner].tolist())
                keep.extend(selected)

    train_df = df.loc[sorted(set(keep))].reset_index(drop=True)

    orig_n   = df[df['patient_id'].isin(train_pids)].shape[0]
    print(f"\nEpisode sampling (train only, {rows_per_episode} rows/episode):")
    print(f"  Before: {orig_n:,} rows → After: {len(train_df):,} rows")
    vc = train_df[TARGET_COL].value_counts().sort_index()
    for cls, cnt in vc.items():
        print(f"    {CLASS_NAMES[int(cls)]:9s}: {cnt:5,}  ({cnt/len(train_df)*100:.1f}%)")
    return train_df


# ════════════════════════════════════════════════════════════
# 5.  FEATURE COLUMNS
# ════════════════════════════════════════════════════════════
def get_feature_cols(df):
    # Never include these
    exclude = {
        'patient_id', 'time',
        TARGET_COL,
        'result_label',      # current FSM state — directly correlated with target
        'severity_label',    # same
        'resp_rate_smoothed',# redundant with resp_rate
        'age',               # constant for all patients
        'selected_cond_1', 'selected_cond_2',
        'time_in_state_min', 'time_in_state_steps',
    }

    cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().mean() > 0.95
    ]

    # Group by type for summary
    raw   = [c for c in cols if c in ['spo2','heart_rate','resp_rate','sbp',
                                       'dbp','mbp','etco2','pulse_pressure']]
    zs    = [c for c in cols if c.startswith('z_') and 'slope' not in c]
    ss    = [c for c in cols if c.startswith('s_') and 'slope' not in c]
    flags = [c for c in cols if c in ['severity_sum','combined_score',
                                       'any_vital_E','any_vital_C',
                                       'tier1_any','tier2_any',
                                       't3_hyper_emergency','t3_stable_deceiver',
                                       't3_masked_shock','t3_occult_acidosis',
                                       't3_trend_decline']]
    s2m   = [c for c in cols if 'slope_2m'  in c]
    s5m   = [c for c in cols if 'slope_5m'  in c]
    s7m   = [c for c in cols if 'slope_7m'  in c and 'accel' not in c]
    s15m  = [c for c in cols if 'slope_15m' in c]
    acc   = [c for c in cols if 'accel'  in c]
    mns   = [c for c in cols if 'mean_'  in c or 'std_' in c]
    crs   = [c for c in cols if 'cross_' in c]

    print(f"\nFeature summary:")
    print(f"  Raw vitals       : {len(raw):4d}")
    print(f"  Z-scores         : {len(zs):4d}")
    print(f"  S-scores         : {len(ss):4d}")
    print(f"  Flags/scores     : {len(flags):4d}")
    print(f"  Slopes 2m        : {len(s2m):4d}  ← already in dataset")
    print(f"  Slopes 5m        : {len(s5m):4d}  ← already in dataset")
    print(f"  Slopes 7m        : {len(s7m):4d}  ← already in dataset")
    print(f"  Slopes 15m       : {len(s15m):4d}  ← added here")
    print(f"  Acceleration     : {len(acc):4d}  ← added here")
    print(f"  Mean/std rolling : {len(mns):4d}  ← added here")
    print(f"  Cross-vital      : {len(crs):4d}  ← added here")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL            : {len(cols):4d}")
    return cols


# ════════════════════════════════════════════════════════════
# 6.  BUILD DATASET
# ════════════════════════════════════════════════════════════
def build_dataset(df, patient_ids, feat_cols, label=''):
    mask = df['patient_id'].isin(patient_ids)
    sub  = df[mask]
    X    = sub[feat_cols].fillna(0).values.astype(np.float32)
    y    = sub[TARGET_COL].values.astype(np.int64)
    no   = (y == 0).mean() * 100
    cr   = (y == 1).mean() * 100
    em   = (y == 2).mean() * 100
    print(f"  {label:6}: {len(y):,} samples | "
          f"N={no:.1f}%  Cr={cr:.1f}%  Em={em:.1f}%")
    return X, y


# ════════════════════════════════════════════════════════════
# 7.  TRAIN
# ════════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, X_val, y_val):
    print(f"\nTraining XGBoost...")
    print(f"  Train : {X_train.shape[0]:,} samples | {X_train.shape[1]} features")
    print(f"  Val   : {X_val.shape[0]:,} samples")
    print(f"  Class weights: {CLASS_WEIGHTS}")

    sw = np.array([CLASS_WEIGHTS.get(int(l), 1.0) for l in y_train])

    model = XGBClassifier(
        n_estimators          = 3000,
        max_depth             = 4,
        learning_rate         = 0.02,
        subsample             = 0.7,
        colsample_bytree      = 0.6,
        min_child_weight      = 10,
        gamma                 = 0.3,
        reg_alpha             = 1.0,
        reg_lambda            = 5.0,
        use_label_encoder     = False,
        eval_metric           = 'mlogloss',
        early_stopping_rounds = 150,
        random_state          = SEED,
        n_jobs                = -1,
    )

    model.fit(
        X_train, y_train,
        sample_weight = sw,
        eval_set      = [(X_val, y_val)],
        verbose       = 200,
    )
    print(f"  Best iteration: {model.best_iteration}")
    if model.best_iteration < 50:
        print(f"  ⚠ best_iteration very low — model may be overfitting fast.")
        print(f"    Try increasing reg_lambda or reducing max_depth.")
    return model


# ════════════════════════════════════════════════════════════
# 8.  EVALUATE
# ════════════════════════════════════════════════════════════
def evaluate(model, X, y, name=''):
    probs = model.predict_proba(X)
    preds = probs.argmax(axis=1)
    f1    = f1_score(y, preds, average='macro', zero_division=0)
    pc    = f1_score(y, preds, average=None, labels=[0,1,2], zero_division=0)

    print(f"\n{'='*60}")
    print(f"{name} — {len(y):,} samples")
    print(f"{'='*60}")
    print(classification_report(y, preds, target_names=CLASS_NAMES,
                                 zero_division=0))
    try:
        auc = roc_auc_score(y, probs, multi_class='ovr', average='macro')
        print(f"AUC-ROC (macro) : {auc:.4f}")
    except Exception:
        pass

    cc = int(((preds == 1) & (y == 1)).sum())
    ct = int((y == 1).sum())
    print(f"Critical found  : {cc}/{ct} ({cc / max(ct,1) * 100:.1f}%)")
    print(f"Macro F1        : {f1:.4f}  "
          f"(N={pc[0]:.3f}  Cr={pc[1]:.3f}  Em={pc[2]:.3f})")
    return preds, probs, f1


# ════════════════════════════════════════════════════════════
# 9.  PLOTS
# ════════════════════════════════════════════════════════════
def plot_confusion(y, preds, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y, preds),
                           display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def plot_distributions(y_train, y_val, y_test, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (y, title) in zip(axes, [
        (y_train, f'Train (episoded)\n{len(y_train):,}'),
        (y_val,   f'Val (all rows)\n{len(y_val):,}'),
        (y_test,  f'Test (all rows)\n{len(y_test):,}'),
    ]):
        counts = [(y == i).sum() for i in range(3)]
        bars   = ax.bar(CLASS_NAMES, counts,
                        color=['#2ecc71', '#e74c3c', '#e67e22'])
        ax.set_title(title)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    f'{cnt:,}', ha='center', fontsize=8)
    plt.suptitle('Class Distributions (Stratified Split)')
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def plot_importance(model, feat_names, path, top_n=40):
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 12))
    cmap = {
        'slope_2m' : '#ff5252', 'slope_5m' : '#ff9800',
        'slope_7m' : '#e67e22', 'slope_15m': '#3498db',
        'accel'    : '#1abc9c',
        'cross_'   : '#e91e63', 'mean_'    : '#607d8b',
        'std_'     : '#455a64',
    }
    colors = []
    for name in np.array(feat_names)[idx]:
        c = '#95a5a6'
        for k, v in cmap.items():
            if k in name: c = v; break
        colors.append(c)
    ax.barh(np.array(feat_names)[idx], imp[idx], color=colors)
    ax.set_title(f'Top {top_n} Features by Memory Scale')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in cmap.items()],
              loc='lower right', fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved → {path}")


def plot_shap(model, X, feat_names, path):
    print("\nSHAP analysis...")
    n    = min(500, len(X))
    expl = shap.TreeExplainer(model)
    sv   = expl.shap_values(X[:n])
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ci, cls in enumerate(CLASS_NAMES):
        plt.sca(axes[ci])
        sv_c = sv[ci] if isinstance(sv, list) else sv
        shap.summary_plot(sv_c, X[:n], feature_names=feat_names,
                          show=False, max_display=15, plot_type='bar')
        axes[ci].set_title(f'SHAP — {cls}')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved → {path}")


# ════════════════════════════════════════════════════════════
# 10.  SAVE / PREDICT
# ════════════════════════════════════════════════════════════
def save_artifacts(model, scaler, feat_cols,
                   mp='vitals_model.json',
                   sp='vitals_scaler.pkl',
                   fp='vitals_features.pkl'):
    model.save_model(mp)
    joblib.dump(scaler,    sp)
    joblib.dump(feat_cols, fp)
    print(f"\nSaved → {mp}, {sp}, {fp}")


def predict_new_patient(patient_df, feat_cols,
                        mp='vitals_model.json',
                        sp='vitals_scaler.pkl'):
    """
    Predict 20-min condition for a NEW unseen patient.

    patient_df must have been through:
      1. catevcode.py       → z-scores, s-scores, severity, FSM
      2. add_extended_memory() from this file → 15m slopes, accel, score_momentum

    Works from row 1 (expanding_slope = no warmup).
    Accuracy improves as more history accumulates.
    """
    model  = XGBClassifier(); model.load_model(mp)
    scaler = joblib.load(sp)

    row    = (patient_df[feat_cols]
              .iloc[[-1]].fillna(0).values.astype(np.float32))
    probs  = model.predict_proba(scaler.transform(row))[0]
    pred   = CLASS_NAMES[int(probs.argmax())]
    hist   = len(patient_df) * 2 / 60
    note   = (f"  ⚠ {hist:.1f} min history — slopes not yet fully populated"
              if hist < 7 else "")

    print(f"\n  20-min prediction: {pred}{note}")
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        bar   = '█' * int(p * 40)
        alert = (' ← ALERT'          if i == 1 and p > 0.4 else
                 ' ← CRITICAL ALERT' if i == 2 and p > 0.3 else '')
        print(f"    {cls:9s}: {p:.3f}  {bar}{alert}")
    return pred, dict(zip(CLASS_NAMES, probs.tolist()))


# ════════════════════════════════════════════════════════════
# 11.  MAIN
# ════════════════════════════════════════════════════════════
def main():
    FILEPATH         = 'your_data.csv'   # ← change this to your CSV path
    ROWS_PER_EPISODE = 5                 # rows kept per FSM label run
                                         # lower = less overfitting, fewer samples
                                         # try 3 if model still collapses
    SAVE             = True

    # 1. Load
    df = load_data(FILEPATH)

    # 2. Add 15m slopes + acceleration
    df = add_extended_memory(df)

    # 3. Feature columns
    feat_cols = get_feature_cols(df)

    # 4. Stratified patient split
    train_pids, val_pids, test_pids = split_patients_stratified(df)

    # 5. Episode sample TRAIN only
    train_df = episode_sample(df, train_pids, ROWS_PER_EPISODE)

    # 6. Build datasets
    print("\nDatasets:")
    X_train, y_train = build_dataset(train_df, train_pids, feat_cols, 'Train')
    X_val,   y_val   = build_dataset(df,       val_pids,   feat_cols, 'Val')
    X_test,  y_test  = build_dataset(df,       test_pids,  feat_cols, 'Test')

    # 7. Scale — fit on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    plot_distributions(y_train, y_val, y_test, 'label_distribution.png')

    # 8. Train
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # 9. Evaluate
    vp, _, vf1 = evaluate(model, X_val,  y_val,  'Validation')
    tp, _, tf1 = evaluate(model, X_test, y_test, 'Test')

    plot_confusion(y_val,  vp, 'Validation Confusion Matrix', 'val_confusion.png')
    plot_confusion(y_test, tp, 'Test Confusion Matrix',       'test_confusion.png')
    plot_importance(model, feat_cols, 'feature_importance.png')
    plot_shap(model, X_test, feat_cols, 'shap_all_classes.png')

    # 10. Save
    if SAVE:
        save_artifacts(model, scaler, feat_cols)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Val  Macro F1 : {vf1:.4f}")
    print(f"  Test Macro F1 : {tf1:.4f}")
    print(f"\nIf Critical F1 is low, adjust:")
    print(f"  CLASS_WEIGHTS[1] = 8.0   (stronger Critical weight)")
    print(f"  ROWS_PER_EPISODE = 3     (more aggressive deduplication)")

    return model, scaler, feat_cols


if __name__ == '__main__':
    model, scaler, feat_cols = main()
