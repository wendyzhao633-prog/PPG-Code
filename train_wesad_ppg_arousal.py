import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Root folder containing the unpacked WESAD dataset.
WESAD_ROOT = r"C:\Users\User\OneDrive\桌面\WESAD\WESAD"

# WESAD subject IDs to process (skip S1 & S12 per the original README).
SUBJECT_IDS = [
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
]

FS_WRIST = 64.0
FS_LABEL = 700.0
WINDOW_SEC = 10
STEP_SEC = 5
WINDOW_SAMPLES = int(WINDOW_SEC * FS_WRIST)
STEP_SAMPLES = int(STEP_SEC * FS_WRIST)

# Default arousal targets if questionnaires are missing.
DEFAULT_AROUSAL = {
    1: 0.1,  # baseline
    2: 0.9,  # stress
    3: 0.6,  # amusement
    4: 0.2,  # meditation
}

PHASE_ORDER = ["Base", "TSST", "Medi 1", "Fun", "Medi 2"]


def _normalize_sam(score: float) -> float:
    """Map SAM 1-9 scale to 0-1 interval."""
    return max(0.0, min(1.0, (score - 1.0) / 8.0))


def load_subject_arousal_scores(sid: str) -> Dict[int, float]:
    """Load per-subject SAM arousal ratings from the questionnaire file."""
    quest_path = os.path.join(WESAD_ROOT, sid, f"{sid}_quest.csv")
    if not os.path.isfile(quest_path):
        return {}

    arousal_values: List[float] = []
    try:
        with open(quest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("# DIM;"):
                    parts = [p for p in line.strip().split(";") if p]
                    if len(parts) >= 3:
                        try:
                            arousal_values.append(float(parts[2]))
                        except ValueError:
                            continue
    except OSError:
        return {}

    if not arousal_values:
        return {}

    score_map: Dict[int, float] = {}
    # Map baseline/stress/amusement.
    if len(arousal_values) >= 1:
        score_map[1] = _normalize_sam(arousal_values[0])
    if len(arousal_values) >= 2:
        score_map[2] = _normalize_sam(arousal_values[1])
    if len(arousal_values) >= 4:
        score_map[3] = _normalize_sam(arousal_values[3])

    # Meditation: prefer the last available med entry.
    med_scores = []
    if len(arousal_values) >= 3:
        med_scores.append(arousal_values[2])
    if len(arousal_values) >= 5:
        med_scores.append(arousal_values[4])
    if med_scores:
        score_map[4] = _normalize_sam(float(np.mean(med_scores)))

    return score_map


def bandpass_filter(sig: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def sliding_windows(sig: np.ndarray, window_samples: int, step_samples: int) -> Sequence[Tuple[int, np.ndarray]]:
    starts = np.arange(0, len(sig) - window_samples + 1, step_samples)
    for start in starts:
        yield int(start), sig[start : start + window_samples]


def detect_peaks_ppg(ppg: np.ndarray, fs: float) -> np.ndarray:
    min_distance = int(0.25 * fs)  # ignore peaks closer than 240 bpm
    peaks, _ = find_peaks(ppg, distance=min_distance)
    return peaks


def hrv_features_from_peaks(peaks: np.ndarray, fs: float) -> Optional[Dict[str, float]]:
    if len(peaks) < 3:
        return None

    rr_intervals = np.diff(peaks) / fs
    if np.any(rr_intervals <= 0):
        return None

    hr = 60.0 / rr_intervals
    hr_mean = float(np.mean(hr))
    hr_std = float(np.std(hr))
    sdnn = float(np.std(rr_intervals))

    diff_rr = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(diff_rr**2))) if len(diff_rr) > 0 else 0.0

    lf = hf = lf_hf = 0.0
    if len(rr_intervals) > 4:
        fs_rr = 1.0 / np.mean(rr_intervals)
        rr_detrend = rr_intervals - np.mean(rr_intervals)
        f, pxx = welch(rr_detrend, fs=fs_rr, nperseg=min(256, len(rr_detrend)))
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        lf_mask = (f >= lf_band[0]) & (f < lf_band[1])
        hf_mask = (f >= hf_band[0]) & (f < hf_band[1])
        lf = float(np.trapz(pxx[lf_mask], f[lf_mask]))
        hf = float(np.trapz(pxx[hf_mask], f[hf_mask]))
        if hf > 1e-6:
            lf_hf = float(lf / hf)

    return {
        "HR_mean": hr_mean,
        "HR_std": hr_std,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "LF": lf,
        "HF": hf,
        "LF_HF": lf_hf,
    }


def ppg_morph_features(ppg: np.ndarray) -> Dict[str, float]:
    return {
        "PPG_amp_mean": float(np.mean(ppg)),
        "PPG_amp_std": float(np.std(ppg)),
        "PPG_amp_range": float(np.max(ppg) - np.min(ppg)),
    }


def extract_features_for_window(ppg_window: np.ndarray, fs: float) -> Optional[Dict[str, float]]:
    ppg_filt = bandpass_filter(ppg_window, fs)
    peaks = detect_peaks_ppg(ppg_filt, fs)
    hrv_feats = hrv_features_from_peaks(peaks, fs)
    if hrv_feats is None:
        return None

    feats = dict(hrv_feats)
    feats.update(ppg_morph_features(ppg_filt))
    return feats


def build_dataset_from_wesad() -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, float]]]:
    X: List[List[float]] = []
    y: List[float] = []
    meta: List[Dict[str, float]] = []
    feature_names: Optional[List[str]] = None

    for sid in SUBJECT_IDS:
        pkl_path = os.path.join(WESAD_ROOT, sid, f"{sid}.pkl")
        if not os.path.isfile(pkl_path):
            print(f"[WARN] Missing pickle file: {pkl_path}")
            continue

        print(f"\n=== Processing subject {sid} ===")
        subject_scores = load_subject_arousal_scores(sid)
        if subject_scores:
            print("Using SAM arousal scores:", subject_scores)
        else:
            print("No questionnaire scores found; falling back to defaults.")
        with open(pkl_path, "rb") as handle:
            data = pickle.load(handle, encoding="latin1")

        bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float)
        if bvp.ndim > 1:
            bvp = bvp.squeeze()
        labels = np.asarray(data["label"], dtype=int)

        dur_label = len(labels) / FS_LABEL
        dur_bvp = len(bvp) / FS_WRIST
        print(
            f"{sid}: BVP length={len(bvp)}, Label length={len(labels)}, "
            f"dur_label={dur_label:.1f}s, dur_bvp={dur_bvp:.1f}s"
        )

        for start_idx, win in sliding_windows(bvp, WINDOW_SAMPLES, STEP_SAMPLES):
            t_center = (start_idx + WINDOW_SAMPLES / 2.0) / FS_WRIST
            label_idx = int(round(t_center * FS_LABEL))
            if label_idx < 0 or label_idx >= len(labels):
                continue

            label_val = labels[label_idx]
            if label_val not in DEFAULT_AROUSAL:
                continue

            arousal_val = subject_scores.get(label_val, DEFAULT_AROUSAL[label_val])
            feats = extract_features_for_window(win, FS_WRIST)
            if feats is None:
                continue

            if feature_names is None:
                feature_names = list(feats.keys())

            X.append([feats[name] for name in feature_names])
            y.append(arousal_val)
            meta.append({"subject": sid, "label_id": int(label_val), "t_center": float(t_center)})

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    return X_arr, y_arr, feature_names or [], meta


def train_arousal_regressor(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> RandomForestRegressor:
    print("\nSplitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    model = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    print("\nTraining RandomForestRegressor...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test RMSE: {np.sqrt(mse):.4f}")
    print(f"Test R^2: {r2:.4f}")
    print(f"y_test mean={np.mean(y_test):.3f}, y_pred mean={np.mean(y_pred):.3f}")

    out_path = "ppg_arousal_wesad_model.pkl"
    joblib.dump({"model": model, "feature_names": feature_names}, out_path)
    print(f"\nSaved trained model to {out_path}")

    return model


def main() -> None:
    print("===== Building PPG->Arousal dataset from WESAD =====")
    X, y, feature_names, _ = build_dataset_from_wesad()
    print("\nFeature matrix shape:", X.shape)
    print("Arousal stats: min =", np.min(y), ", max =", np.max(y))

    print("\n===== Training regressor =====")
    train_arousal_regressor(X, y, feature_names)


if __name__ == "__main__":
    main()
