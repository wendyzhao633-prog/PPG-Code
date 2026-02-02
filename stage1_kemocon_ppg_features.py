# stage1_kemocon_ppg_features.py
# Stage 1: K-EmoCon E4_BVP.csv -> window-level PPG features (+ quality) using 10s window / 5s hop

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch


# -----------------------------
# Utils
# -----------------------------
def robust_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV robustly (handles Excel-like formatting)."""
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    required = {"timestamp", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Found: {df.columns.tolist()}")
    return df


def to_seconds(ts: np.ndarray) -> np.ndarray:
    """
    Convert timestamps to seconds.
    Heuristic:
    - if values > 1e12 => assume epoch milliseconds
    - if values > 1e9  => assume epoch seconds
    - else assume already seconds
    """
    ts = ts.astype(np.float64)
    m = np.nanmedian(ts)
    if m > 1e12:
        return ts / 1000.0
    if m > 1e9:
        return ts
    return ts


def estimate_fs(t_sec: np.ndarray, default_fs: float = 64.0) -> float:
    """Estimate sampling rate from median dt, with fallback."""
    t = np.asarray(t_sec, dtype=np.float64)
    t = t[np.isfinite(t)]
    if len(t) < 10:
        return default_fs
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) < 10:
        return default_fs
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        return default_fs
    fs = 1.0 / med_dt
    # Clamp to reasonable range for E4 BVP
    if fs < 10 or fs > 200:
        return default_fs
    return float(fs)


def bandpass_filter(sig: np.ndarray, fs: float, low: float = 0.5, high: float = 3.0, order: int = 3) -> np.ndarray:
    """Bandpass filter for PPG/BVP."""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    if high_n >= 1.0:
        high_n = 0.99
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, sig)


@dataclass
class WindowFeatures:
    hr_mean: float
    hr_std: float
    sdnn: float
    rmssd: float
    lf: float
    hf: float
    lf_hf: float
    ppg_amp_mean: float
    ppg_amp_std: float
    ppg_amp_range: float
    n_peaks: int
    peak_prominence_mean: float
    mask_available: int


def compute_features_one_window(ppg_win: np.ndarray, fs: float) -> WindowFeatures:
    """
    Compute 10-dim physiological features + quality fields.
    Uses peak detection on filtered signal.
    """
    x = np.asarray(ppg_win, dtype=np.float64)
    if len(x) < int(fs * 2):  # too short
        return WindowFeatures(*(np.nan,) * 10, 0, 0.0, 0)

    xf = bandpass_filter(x, fs)

    # Peak detection
    min_distance = max(1, int(0.4 * fs))  # 0.4s
    prom_thr = 0.3 * float(np.std(xf) + 1e-9)
    peaks, props = find_peaks(xf, distance=min_distance, prominence=prom_thr)

    n_peaks = int(len(peaks))
    prom_mean = float(np.mean(props["prominences"])) if n_peaks > 0 and "prominences" in props else 0.0

    # Availability rule (MVP)
    if n_peaks < 3:
        return WindowFeatures(*(np.nan,) * 10, n_peaks, prom_mean, 0)

    rr = np.diff(peaks) / fs  # seconds
    rr = rr[(rr > 0.25) & (rr < 2.0)]  # plausible RR: 30–240 bpm
    if len(rr) < 2:
        return WindowFeatures(*(np.nan,) * 10, n_peaks, prom_mean, 0)

    hr_inst = 60.0 / rr
    hr_mean = float(np.mean(hr_inst))
    hr_std = float(np.std(hr_inst))

    sdnn = float(np.std(rr)) if len(rr) > 1 else np.nan
    rr_diff = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else np.nan

    # LF/HF (can be unstable for short windows; keep but treat cautiously)
    lf = hf = lf_hf = 0.0
    if len(rr) >= 5:
        t_rr = np.cumsum(rr)
        dur = float(t_rr[-1] - t_rr[0])
        if dur > 0:
            # Interpolate RR to 4 Hz
            t_uniform = np.linspace(t_rr[0], t_rr[-1], int(dur * 4) + 1)
            rr_interp = np.interp(t_uniform, t_rr, rr)
            f, pxx = welch(rr_interp, fs=4.0, nperseg=min(256, len(rr_interp)))

            lf_band = (0.04, 0.15)
            hf_band = (0.15, 0.40)
            lf_mask = (f >= lf_band[0]) & (f < lf_band[1])
            hf_mask = (f >= hf_band[0]) & (f < hf_band[1])

            lf = float(np.trapz(pxx[lf_mask], f[lf_mask])) if np.any(lf_mask) else 0.0
            hf = float(np.trapz(pxx[hf_mask], f[hf_mask])) if np.any(hf_mask) else 0.0
            lf_hf = float(lf / hf) if hf > 0 else 0.0

    # Amplitude stats on filtered signal
    ppg_amp_mean = float(np.mean(xf))
    ppg_amp_std = float(np.std(xf))
    ppg_amp_range = float(np.max(xf) - np.min(xf))

    return WindowFeatures(
        hr_mean=hr_mean,
        hr_std=hr_std,
        sdnn=sdnn,
        rmssd=rmssd,
        lf=lf,
        hf=hf,
        lf_hf=lf_hf,
        ppg_amp_mean=ppg_amp_mean,
        ppg_amp_std=ppg_amp_std,
        ppg_amp_range=ppg_amp_range,
        n_peaks=n_peaks,
        peak_prominence_mean=prom_mean,
        mask_available=1,
    )


def window_iter(t: np.ndarray, x: np.ndarray, window_sec: float, hop_sec: float) -> List[Tuple[float, float, np.ndarray]]:
    """
    Create windows by time (not by index), robust to small timestamp jitter.
    Returns list of (t_start, t_end, samples).
    """
    t0 = float(t[0])
    t_end_all = float(t[-1])
    out = []
    start = t0
    while start + window_sec <= t_end_all:
        end = start + window_sec
        mask = (t >= start) & (t < end)
        seg = x[mask]
        if len(seg) > 0:
            out.append((start - t0, end - t0, seg))
        start += hop_sec
    return out


def process_one_session(bvp_csv: Path, session_id: str, window_sec: float, hop_sec: float, default_fs: float) -> pd.DataFrame:
    df = robust_read_csv(bvp_csv)

    # Parse time + signal
    t_sec = to_seconds(df["timestamp"].to_numpy())
    x = df["value"].to_numpy(dtype=np.float64)

    # Sort by time (important)
    order = np.argsort(t_sec)
    t_sec = t_sec[order]
    x = x[order]

    # Remove NaNs
    ok = np.isfinite(t_sec) & np.isfinite(x)
    t_sec = t_sec[ok]
    x = x[ok]
    if len(x) < 100:
        return pd.DataFrame()

    fs = estimate_fs(t_sec, default_fs=default_fs)

    rows: List[Dict[str, Any]] = []
    for t_start_rel, t_end_rel, seg in window_iter(t_sec, x, window_sec=window_sec, hop_sec=hop_sec):
        feat = compute_features_one_window(seg, fs=fs)
        rows.append({
            "session_id": session_id,
            "fs_est": fs,
            "t_start_sec": round(t_start_rel, 3),
            "t_end_sec": round(t_end_rel, 3),
            # 10-dim physiological
            "HR_mean": feat.hr_mean,
            "HR_std": feat.hr_std,
            "SDNN": feat.sdnn,
            "RMSSD": feat.rmssd,
            "LF": feat.lf,
            "HF": feat.hf,
            "LF_HF": feat.lf_hf,
            "PPG_amp_mean": feat.ppg_amp_mean,
            "PPG_amp_std": feat.ppg_amp_std,
            "PPG_amp_range": feat.ppg_amp_range,
            # quality fields
            "n_peaks": feat.n_peaks,
            "peak_prominence_mean": feat.peak_prominence_mean,
            "mask_available": feat.mask_available,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4-root", required=True, help="Path to extracted e4_data folder (contains 1/,2/,3/... folders)")
    ap.add_argument("--out", default="kemocon_ppg_features.csv", help="Output CSV path")
    ap.add_argument("--window-sec", type=float, default=10.0, help="Window length in seconds")
    ap.add_argument("--hop-sec", type=float, default=5.0, help="Hop length in seconds")
    ap.add_argument("--default-fs", type=float, default=64.0, help="Fallback sampling rate if fs estimation fails")
    args = ap.parse_args()

    e4_root = Path(args.e4_root)
    if not e4_root.exists():
        raise FileNotFoundError(e4_root)

    all_rows = []
    # Each subfolder like e4_data/1/, e4_data/2/...
    for sub in sorted([p for p in e4_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        bvp = sub / "E4_BVP.csv"
        if not bvp.exists():
            continue
        session_id = sub.name  # use folder name as session_id
        print(f"Processing session {session_id}: {bvp}")
        df_feat = process_one_session(
            bvp_csv=bvp,
            session_id=session_id,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            default_fs=args.default_fs,
        )
        if len(df_feat) > 0:
            all_rows.append(df_feat)

    if not all_rows:
        print("No sessions processed. Check folder structure and file names.")
        return

    out_df = pd.concat(all_rows, ignore_index=True)
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path.resolve()}")
    print(out_df.head())


if __name__ == "__main__":
    main()
