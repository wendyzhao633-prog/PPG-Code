import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

try:
    import joblib
except ImportError:  # fallback when joblib is not installed
    joblib = None
    import pickle

# Ensure the console can print UTF-8 (fixes Windows cp1252 errors when printing Chinese)
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# =============== 基本参数（根据你现在的设置） =================
CSV_PATH = "Test_Deepbreath.csv" # 这里改成你的 csv 文件名
MODEL_PATH = "ppg_arousal_wesad_model.pkl"  # 之前训练好的模型
FS = 64                                     # 你已经把 Sample Rate 设成 64 sps
WINDOW_SEC = 10                             # 每个窗口长度
STEP_SEC = 5                                # 窗口滑动步长（重叠一半）

WINDOW_SAMPLES = int(WINDOW_SEC * FS)
STEP_SAMPLES = int(STEP_SEC * FS)


# =============== 一些工具函数（和训练时类似） =================
def bandpass_filter(sig: np.ndarray, fs: int, low: float = 0.5, high: float = 3.0, order: int = 3):
    """对 PPG 做 0.5–3 Hz 带通滤波，聚焦正常静息/轻运动心率范围，抑制高频假峰。"""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def compute_hrv_and_ppg_features(ppg_win: np.ndarray, fs: int) -> np.ndarray:
    """
    对一个窗口的 PPG 计算 10 个特征：
    1. mean HR
    2. HR std
    3. SDNN
    4. RMSSD
    5. LF power
    6. HF power
    7. LF/HF ratio
    8. PPG mean
    9. PPG std
    10. PPG peak-to-peak (max-min)
    """
    # 先带通滤波
    ppg_f = bandpass_filter(ppg_win, fs)

    # 找心跳峰：放宽到至少 0.5 s 间隔（对应 HR <= 120 bpm）并要求一定凸起幅度
    min_distance = max(1, int(0.5 * fs))
    prominence = 0.3 * np.std(ppg_f)  # 相对幅度阈值，抑制杂峰
    peaks, _ = find_peaks(ppg_f, distance=min_distance, prominence=prominence)

    # 如果峰太少，说明这个窗口质量不好，返回 NaN
    if len(peaks) < 3:
        return np.full(10, np.nan, dtype=float)

    # 计算 RR 间期（秒）
    rr = np.diff(peaks) / fs  # in seconds

    # 心率（bpm）
    rr_mean = np.mean(rr)
    hr = 60.0 / rr_mean if rr_mean > 0 else np.nan
    hr_std = np.std(60.0 / rr) if len(rr) > 1 else np.nan

    # SDNN & RMSSD
    sdnn = np.std(rr) if len(rr) > 1 else np.nan
    rr_diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else np.nan

    # 简单的 LF / HF 功率（对 RR 序列做 Welch）
    if len(rr) > 4:
        # 为了避免采样率太低，这里简单把 RR 插值到 4 Hz
        t_rr = np.cumsum(rr) - rr[0]
        t_uniform = np.linspace(t_rr[0], t_rr[-1], int((t_rr[-1] - t_rr[0]) * 4) + 1)
        rr_interp = np.interp(t_uniform, t_rr, rr)

        f, pxx = welch(rr_interp, fs=4.0, nperseg=min(256, len(rr_interp)))
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.40)
        lf_mask = (f >= lf_band[0]) & (f < lf_band[1])
        hf_mask = (f >= hf_band[0]) & (f < hf_band[1])
        lf = float(np.trapz(pxx[lf_mask], f[lf_mask])) if np.any(lf_mask) else 0.0
        hf = float(np.trapz(pxx[hf_mask], f[hf_mask])) if np.any(hf_mask) else 0.0
        lf_hf = lf / hf if hf > 0 else 0.0
    else:
        lf = hf = lf_hf = 0.0

    # PPG 振幅特征
    amp_mean = float(np.mean(ppg_f))
    amp_std = float(np.std(ppg_f))
    amp_pp = float(np.max(ppg_f) - np.min(ppg_f))

    feats = np.array(
        [hr, hr_std, sdnn, rmssd, lf, hf, lf_hf, amp_mean, amp_std, amp_pp],
        dtype=float,
    )
    return feats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict arousal from MAX86140 PPG CSV.")
    parser.add_argument("--csv", dest="csv_path", default=CSV_PATH, help="Path to input CSV")
    parser.add_argument("--model", dest="model_path", default=MODEL_PATH, help="Path to trained model pickle")
    parser.add_argument("--fs", dest="fs", type=int, default=FS, help="Sampling rate of PPG signal (Hz)")
    parser.add_argument("--no-save", action="store_true", help="只打印，不写 CSV")
    return parser.parse_args()


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if joblib is not None:
        return joblib.load(model_path)
    # joblib 未安装时，改用内置 pickle 加载模型
    import pickle

    with model_path.open("rb") as f:
        return pickle.load(f)


def read_ppg_csv(csv_path: Path) -> pd.DataFrame:
    """
    读取 MAX86140 CSV；若文件前面有几行寄存器说明（常见在 MAX86140 dump），
    自动尝试用第 6 行作为表头（header=5）。
    """
    df = pd.read_csv(csv_path)
    candidate_cols = {"LEDC1", "LED1", "PPG", "PPG1"}
    if candidate_cols.intersection(df.columns):
        return df

    # 常见 MAX86140 导出格式：前几行是寄存器名/说明，后面才是正式列名
    for header_row in (5, 6):
        try:
            df_alt = pd.read_csv(csv_path, header=header_row)
            if candidate_cols.intersection(df_alt.columns):
                return df_alt
        except Exception:
            continue

    return df


# =============== 主流程：读取 CSV -> 提取特征 -> 预测 ===============
def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    model_path = Path(args.model_path)
    fs = args.fs
    window_samples = int(WINDOW_SEC * fs)
    step_samples = int(STEP_SEC * fs)

    print(">>> Running prediction script...")

    print("=== 1. 读取 CSV 数据 ===")
    if not csv_path.exists():
        print(f"❌ 找不到输入 CSV: {csv_path}")
        return
    df = read_ppg_csv(csv_path)

    # 找 PPG 列：优先用 LEDC1
    ppg_col = None
    for cand in ["LEDC1", "LED1", "PPG", "PPG1"]:
        if cand in df.columns:
            ppg_col = cand
            break

    if ppg_col is None:
        raise ValueError("找不到 PPG 列（LEDC1 / LED1 / PPG），请打印 df.columns 看看有哪些列。")

    print(f"使用列 '{ppg_col}' 作为 PPG 信号")
    ppg = pd.to_numeric(df[ppg_col], errors="coerce").dropna().to_numpy(dtype=float)

    print(f"PPG length = {len(ppg)} samples, duration ≈ {len(ppg)/fs:.1f} s")

    print("=== 2. 按 10 秒窗口提取特征 ===")
    features = []
    win_times = []

    start = 0
    while start + window_samples <= len(ppg):
        win = ppg[start: start + window_samples]
        feats = compute_hrv_and_ppg_features(win, fs)

        if not np.any(np.isnan(feats)):
            features.append(feats)
            # 记录这个窗口的起止时间（秒）
            t_start = start / fs
            t_end = (start + window_samples) / fs
            win_times.append((t_start, t_end))

        start += step_samples

    features = np.asarray(features)
    print(f"有效窗口数: {features.shape[0]}, 每个窗口特征维度: {features.shape[1]}")

    if features.shape[0] == 0:
        print("❌ 没有任何有效窗口（可能波形质量差或者时间太短）")
        return

    print("=== 3. 加载训练好的模型，预测 arousal ===")
    try:
        data = load_model(model_path)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return
    model = data["model"]

    y_pred = model.predict(features)
    hr_mean = features[:, 0]  # 第 1 个特征是 mean HR (bpm)

    print("Timestamp(s)\tHR(bpm)\tarousal")
    for i in range(len(y_pred)):
        print(
            f"{win_times[i][0]:6.1f}-{win_times[i][1]:-6.1f}\t"
            f"{hr_mean[i]:6.2f}\t{y_pred[i]:.3f}"
        )

    if args.no_save:
        print("（按 --no-save 选项，不写 CSV）")
        return

    print("=== 4. 保存结果到 CSV ===")
    out_df = pd.DataFrame(
        {
            "t_start_sec": [t[0] for t in win_times],
            "t_end_sec": [t[1] for t in win_times],
            "hr_bpm": hr_mean,
            "arousal_pred": y_pred,
        }
    )
    out_name = "max86140_arousal_pred.csv"
    try:
        out_df.to_csv(out_name, index=False)
        print(f"已保存到 {out_name}")
    except PermissionError:
        alt_name = f"max86140_arousal_pred_{datetime.now():%Y%m%d_%H%M%S}.csv"
        out_df.to_csv(alt_name, index=False)
        print(f"⚠️ 原文件可能被占用，改写到 {alt_name}")


if __name__ == "__main__":
    main()
