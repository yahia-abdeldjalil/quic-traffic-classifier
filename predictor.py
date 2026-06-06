# quic-backend/app/predictor.py

import os
import time
import joblib
import numpy as np
import pandas as pd

from typing import Dict
import tempfile
import numpy as np
import pyshark
from fastapi import UploadFile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACT_PATH = os.path.join(
    BASE_DIR,
    "..",
    "ml",
    "artifacts",
    # "quic_ensemble.joblib",
    # "claude_v1",
    #"quic_classifier_vclaude_v4.joblib",
    #"current", "quic_classifier_vclaude_v5.joblib",
    # "current", "quic_ensemble.joblib"
    "kaggle", "quic_classifier_vclaude_v5.joblib"
)

print(f"[INFO] Loading artifact from: {ARTIFACT_PATH}")
start = time.perf_counter()
artifact = joblib.load(ARTIFACT_PATH)
end = time.perf_counter()
print(
f"[INFO] Artifact loaded in "
f"{end - start:.2f} seconds"
)
print("STORED ARTIFACT KEYS:", list(artifact.keys()))
MODELS = artifact["models"]
SCALER = artifact["scaler"]
FEATURE_NAMES = artifact["feature_names"]
GA_WEIGHTS = artifact.get("ga_weights")
PSO_WEIGHTS = artifact.get("pso_weights")
# application id -> name mapping
APP_ID_TO_NAME = {
    0: 'default-background',
    1: 'google-www',
    2: 'google-play',
    3: 'google-services',
    4: 'google-gstatic',
    5: 'google-usercontent',
    6: 'google-authentication',
    7: 'google-conncheck',
    8: 'google-fonts',
    9: 'google-drive',
    10: 'google-docs',
    11: 'google-photos',
    12: 'google-colab',
    13: 'google-translate',
    14: 'gmail',
    15: 'youtube',
    16: 'google-hangouts',
    17: 'google-ads',
    18: 'google-safebrowsing',
    19: 'google-recaptcha',
    20: 'google-pay',
    21: 'google-imasdk',
    22: 'google-calendar',
    23: 'google-autofill',
    24: 'google-scholar',
    25: 'firebase-crashlytics',
    26: 'chrome-remotedesktop',
    27: 'facebook-graph',
    28: 'facebook-gamesgraph',
    29: 'facebook-media',
    30: 'facebook-web',
    31: 'facebook-connect',
    32: 'facebook-rupload',
    33: 'facebook-messenger',
    34: 'instagram',
    35: 'whatsapp',
    36: 'microsoft-outlook',
    37: 'microsoft-substrate',
    38: 'apple-privaterelay',
    39: 'alza-www',
    40: 'alza-webapi',
    41: 'alza-identity',
    42: 'overleaf-compile',
    43: 'overleaf-cdn',
    44: 'tiktok',
    45: 'snapchat',
    46: 'vkontakte',
    47: 'dcard',
    48: 'unitygames',
    49: 'blitz-gg',
    50: 'easybrain',
    51: 'chess-com',
    52: 'poe-ninja',
    53: 'csgo-market',
    54: 'gamedock',
    55: 'playradio',
    56: 'jsdelivr',
    57: 'fontawesome',
    58: 'unpkg',
    59: 'cloudflare-cdnjs',
    60: 'signal-cdn',
    61: 'spanbang',
    62: 'bongacams',
    63: 'xhamster',
    64: 'garmin',
    65: 'dns-doh',
    66: 'flightradar24',
    67: 'endnote-click',
    68: 'fitbit',
    69: 'toggl',
    70: 'doi-org',
    71: 'uber',
    72: 'adavoid',
    73: 'cedexis',
    74: 'bitly',
    75: 'pocasidata-cz',
    76: 'mentimeter',
    77: 'easylist',
    78: 'ncbi-gov',
    79: 'etoro',
    80: 'kaggle',
    81: 'mdpi',
    82: 'livescore',
    83: 'kiwi-com',
    84: 'blogger',
    85: '4chan',
    86: 'forum24',
    87: 'sme-sk',
    88: 'medium',
    89: 'openx',
    90: 'connectad',
    91: 'joinhoney',
    92: 'dm-de',
    93: 'drmax',
    94: 'rohlik',
    95: 'ebay-kleinanzeigen',
    96: 'gothbb',
    97: 'revolut',
    98: 'tawkto',
    99: 'hubspot',
    100: 'goout',
    101: 'spotify',
    102: 'shazam',
    103: 'usercentrics',
    104: 'onesignal',
    105: 'tinypass',
    106: 'discord',
    107: 'hcaptcha',
    108: 'bitdefender-nimbus',
    109: 'google-background',
    110: 'facebook-background'
}
TREE_MODELS = {'XGBoost', 'HistGB', 'RandomForest', 'ExtraTrees'}
print("FEATURES OF TRAINING: ", len(FEATURE_NAMES))
for c in FEATURE_NAMES:
    print(c)
def get_weights(strategy: str):

    strategy = strategy.lower()

    if strategy == "ga":
        return GA_WEIGHTS

    if strategy == "pso":
        return PSO_WEIGHTS

    return (GA_WEIGHTS + PSO_WEIGHTS) / 2
# ======================================================================================
# FEATURE CONSTRUCTION
# ======================================================================================

def construct_row(features: Dict[str, float]):
    """ 
    receives flow parameters 
    returns the full sample (containing coherent values from input features) to infer from
    """
    row = {}

    avg_ipt = features.get("avg_ipt", 50.0)
    std_ipt = features.get("std_ipt", 10.0)

    avg_size = features.get("avg_size", 800.0)
    std_size = features.get("std_size", 100.0)

    ul_dl_ratio_bytes = features.get("ul_dl_ratio_bytes", 1.0)
    ul_dl_ratio_packets = features.get("ul_dl_ratio_packets", 1.0)

    ppi_len = features.get("ppi_len", 30.0)
    ppi_roundtrips = features.get("ppi_roundtrips", 3.0)
    ppi_duration = features.get("ppi_duration", 1000.0) / 1000

    flow_end_reason = int(features.get("flow_end_reason", 0))

    # IPT features
    ipt_values = np.random.normal(avg_ipt, std_ipt, 30)

    for i in range(30):
        row[f"IPT_{i+1}"] = float(max(0.0, ipt_values[i]))

    # DIR features
    for i in range(30):
        row[f"DIR_{i+1}"] = 1.0 if i % 2 == 0 else -1.0

    # SIZE features
    size_values = np.random.normal(avg_size, std_size, 30)

    for i in range(30):
        row[f"SIZE_{i+1}"] = float(max(1.0, size_values[i]))

    # traffic totals
    total_bytes = avg_size * 30

    row["BYTES"] = float(total_bytes * ul_dl_ratio_bytes)
    row["BYTES_REV"] = float(total_bytes)

    row["PACKETS"] = float(30 * ul_dl_ratio_packets)
    row["PACKETS_REV"] = 30.0

    # metadata
    row["DURATION"] = float(ppi_duration)

    row["PPI_LEN"] = float(ppi_len)
    row["PPI_ROUNDTRIPS"] = float(ppi_roundtrips)
    row["PPI_DURATION"] = float(ppi_duration)

    row["FLOW_ENDREASON_IDLE"] = 1.0 if flow_end_reason == 0 else 0.0
    row["FLOW_ENDREASON_ACTIVE"] = 1.0 if flow_end_reason == 1 else 0.0
    row["FLOW_ENDREASON_OTHER"] = 1.0 if flow_end_reason == 2 else 0.0

    # histogram placeholders
    for i in range(1, 9):
        row[f"PSIZE_BIN{i}"] = 0.125
        row[f"PSIZE_BIN{i}_REV"] = 0.125

        row[f"IPT_BIN{i}"] = 0.125
        row[f"IPT_BIN{i}_REV"] = 0.125

    final_row = []

    for feature_name in FEATURE_NAMES:
        final_row.append(row.get(feature_name, 0.0))
    return final_row


def construct_row_v2(features: Dict[str, float]):
    """
    Build a coherent feature row for inference from the provided QUIC flow summary.
    Uses deterministic value construction rather than placeholders.
    """
    row = {}

    avg_ipt = float(features.get("avg_ipt", 50.0))
    std_ipt = float(features.get("std_ipt", 10.0))
    avg_size = float(features.get("avg_size", 800.0))
    std_size = float(features.get("std_size", 100.0))
    ul_dl_ratio_direction = float(features.get("ul_dl_ratio_direction", 0.5))
    ul_dl_ratio_bytes = float(features.get("ul_dl_ratio_bytes", 0.5))
    ul_dl_ratio_packets = float(features.get("ul_dl_ratio_packets", 0.5))
    ppi_len = int(np.clip(int(round(float(features.get("ppi_len", 30.0)))), 2, 30))
    ppi_roundtrips = float(features.get("ppi_roundtrips", 3.0))
    ppi_duration = float(features.get("ppi_duration", 1000.0)) / 1000
    flow_end_reason = int(features.get("flow_end_reason", 0))

    ul_dl_ratio_direction = min(max(ul_dl_ratio_direction, 0.0), 1.0)
    ul_dl_ratio_bytes = min(max(ul_dl_ratio_bytes, 0.0), 1.0)
    ul_dl_ratio_packets = min(max(ul_dl_ratio_packets, 0.0), 1.0)

    def _unit_std_sequence(length: int, phase: float = 0.0):
        if length <= 1:
            return np.array([0.0], dtype=float)
        base = np.cos(np.linspace(phase, phase + 2.0 * np.pi, length, endpoint=False))
        centered = base - np.mean(base)
        std = np.std(centered, ddof=0)
        if std == 0.0:
            return np.zeros(length, dtype=float)
        return centered / std

    # enforce integer and valid ranges
    ppi_len = int(np.clip(int(round(ppi_len)), 2, 30))
    requested_rts = int(max(0, round(ppi_roundtrips)))
    max_possible_rts = ppi_len // 2
    r = int(np.clip(requested_rts, 0, max_possible_rts))

    # ensure direction target is achievable given r (number of + in base pairs = r)
    desired_n_up = int(round(ul_dl_ratio_direction * ppi_len))
    # clamp desired ups so we can distribute remaining packets without creating extra +->- pairs
    desired_n_up = int(np.clip(desired_n_up, r, ppi_len - r))

    m = ppi_len - 2 * r
    extras_up = int(np.clip(desired_n_up - r, 0, m))
    extras_down = int(m - extras_up)

    # build DIR sequence: extras_up (+1), r pairs (+1,-1)*r, extras_down (-1)
    dirs = []
    dirs.extend([1.0] * extras_up)
    for _ in range(r):
        dirs.append(1.0)
        dirs.append(-1.0)
    dirs.extend([-1.0] * extras_down)
    # safety: if length shorter/longer, trim/pad with 0.0
    if len(dirs) < ppi_len:
        dirs.extend([0.0] * (ppi_len - len(dirs)))
    elif len(dirs) > ppi_len:
        dirs = dirs[:ppi_len]

    # create IPT and SIZE sequences of exactly ppi_len values with requested stds
    ipt_offsets = _unit_std_sequence(ppi_len, phase=0.0)
    size_offsets = _unit_std_sequence(ppi_len, phase=0.9)
    active_ipt = (avg_ipt + std_ipt * ipt_offsets).astype(float)
    active_size = (avg_size + std_size * size_offsets).astype(float)

    for i in range(1, 31):
        if i <= ppi_len:
            row[f"IPT_{i}"] = float(active_ipt[i - 1])
            row[f"SIZE_{i}"] = float(active_size[i - 1])
            row[f"DIR_{i}"] = float(dirs[i - 1])
        else:
            row[f"IPT_{i}"] = 0.0
            row[f"SIZE_{i}"] = 0.0
            row[f"DIR_{i}"] = 0.0

    total_bytes = float(np.sum(active_size))
    row["BYTES"] = total_bytes * ul_dl_ratio_bytes
    row["BYTES_REV"] = total_bytes - row["BYTES"]

    total_packets = float(ppi_len)
    row["PACKETS"] = total_packets * ul_dl_ratio_packets
    row["PACKETS_REV"] = total_packets - row["PACKETS"]

    row["DURATION"] = float(ppi_duration)
    row["PPI_LEN"] = float(ppi_len)
    row["PPI_ROUNDTRIPS"] = float(ppi_roundtrips)
    row["PPI_DURATION"] = float(ppi_duration)

    row["FLOW_ENDREASON_IDLE"] = 1.0 if flow_end_reason == 0 else 0.0
    row["FLOW_ENDREASON_ACTIVE"] = 1.0 if flow_end_reason == 1 else 0.0
    row["FLOW_ENDREASON_OTHER"] = 1.0 if flow_end_reason == 2 else 0.0

    def _normalized_distribution(center: float, width: float = 0.18):
        bins = np.linspace(0.05, 0.95, 8)
        scores = np.exp(-0.5 * ((bins - center) / width) ** 2)
        return scores / np.sum(scores)

    size_center = np.clip((avg_size - 64.0) / 1436.0, 0.0, 1.0)
    byte_center = size_center * 0.9 + 0.05
    size_width = 0.12 + 0.08 * (1.0 - ul_dl_ratio_direction)
    psize_bins = _normalized_distribution(byte_center, width=size_width)
    psize_rev_bins = _normalized_distribution(1.0 - byte_center, width=size_width)

    ipt_center = np.clip((avg_ipt - 1.0) / 319.0, 0.0, 1.0)
    ipt_width = 0.12 + 0.08 * ul_dl_ratio_direction
    ipt_bins = _normalized_distribution(ipt_center, width=ipt_width)
    ipt_rev_bins = _normalized_distribution(1.0 - ipt_center, width=ipt_width)

    for i, value in enumerate(psize_bins, start=1):
        row[f"PSIZE_BIN{i}"] = float(value)
    for i, value in enumerate(psize_rev_bins, start=1):
        row[f"PSIZE_BIN{i}_REV"] = float(value)
    for i, value in enumerate(ipt_bins, start=1):
        row[f"IPT_BIN{i}"] = float(value)
    for i, value in enumerate(ipt_rev_bins, start=1):
        row[f"IPT_BIN{i}_REV"] = float(value)

    final_row = [row.get(feature_name, 0.0) for feature_name in FEATURE_NAMES]
    return final_row


def print_construct_row_v2_stats(final_row):
    """Calculate and print summary values from construct_row_v2 output."""
    row = dict(zip(FEATURE_NAMES, final_row))
    print("CONSTRUCT_ROW_V2 OUTPUT:", row)
    ppi_len = int(np.clip(int(round(float(row.get("PPI_LEN", 30.0)))), 2, 30))
    ipt_values = [row.get(f"IPT_{i}", 0.0) for i in range(1, ppi_len + 1)]
    size_values = [row.get(f"SIZE_{i}", 0.0) for i in range(1, ppi_len + 1)]
    dir_values = [row.get(f"DIR_{i}", 0.0) for i in range(1, ppi_len + 1)]

    average_ipt = float(np.mean(ipt_values))
    std_ipt = float(np.std(ipt_values, ddof=0))
    average_size = float(np.mean(size_values))
    std_size = float(np.std(size_values, ddof=0))

    positive_dirs = sum(1 for value in dir_values if value > 0.0)
    ul_dl_direction_ratio = float(positive_dirs / max(1, len(dir_values)))

    bytes_total = row.get("BYTES", 0.0) + row.get("BYTES_REV", 0.0)
    if bytes_total > 0.0:
        ul_dl_byte_ratio = float(row.get("BYTES", 0.0) / bytes_total)
    else:
        ul_dl_byte_ratio = 0.0

    packets_total = row.get("PACKETS", 0.0) + row.get("PACKETS_REV", 0.0)
    if packets_total > 0.0:
        ul_dl_packet_ratio = float(row.get("PACKETS", 0.0) / packets_total)
    else:
        ul_dl_packet_ratio = 0.0

    ppi_duration = float(row.get("PPI_DURATION", 0.0))
    ppi_roundtrips = float(row.get("PPI_ROUNDTRIPS", 0.0))

    if row.get("FLOW_ENDREASON_IDLE", 0.0) == 1.0:
        flow_end_reason = "idle"
    elif row.get("FLOW_ENDREASON_ACTIVE", 0.0) == 1.0:
        flow_end_reason = "active"
    elif row.get("FLOW_ENDREASON_OTHER", 0.0) == 1.0:
        flow_end_reason = "other"
    else:
        flow_end_reason = "unknown"

    print("construct_row_v2 summary:")
    print(f"  Average IPT: {average_ipt:.6f}")
    print(f"  IPT Standard Deviation: {std_ipt:.6f}")
    print(f"  Average Packet Size: {average_size:.6f}")
    print(f"  Packet Size Standard Deviation: {std_size:.6f}")
    print(f"  UL/DL Direction Ratio: {ul_dl_direction_ratio:.6f}")
    print(f"  UL/DL Byte Ratio: {ul_dl_byte_ratio:.6f}")
    print(f"  UL/DL Packet Ratio: {ul_dl_packet_ratio:.6f}")
    print(f"  PPI Duration: {ppi_duration:.6f}")
    print(f"  PPI Length: {ppi_len:.6f}")
    print(f"  PPI Roundtrips: {ppi_roundtrips:.6f}")
    print(f"  Flow End Reason: {flow_end_reason}")

# =====
# Adding features to row input X
# =====
def featurify_157(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # add BYTES_PER_PACKET feature
    X['BYTES_PER_PACKET'] = X['BYTES'] / np.maximum(X['PACKETS'], 1)
    X['BYTES_REV_PER_PKT_REV'] = X['BYTES_REV'] / np.maximum(X['PACKETS_REV'], 1)
    X['PACKET_RATE'] = X['PACKETS'] / np.maximum(X['DURATION'], 1e-6)
    X['PACKET_REV_RATE'] = X['PACKETS_REV'] / np.maximum(X['DURATION'], 1e-6)
    X['DIR_SIZE_RATIO'] = (X['BYTES'] + 1) / (np.maximum(X['BYTES_REV'], 1e-6) + 1)
    X['DIR_PKT_RATIO'] = (X['PACKETS'] + 1) / (X['PACKETS_REV'] + 1)
    _tb = X['BYTES'] + X['BYTES_REV'] + 1
    _tp = X['PACKETS'] + X['PACKETS_REV'] + 1
    X['BYTES_FWD_FRAC'] = X['BYTES'] / _tb
    X['PKTS_FWD_FRAC'] = X['PACKETS'] / _tp
    # Bin-derived features on RAW counts (before log1p)
    def _bin_entropy(m):
        p = m + 1e-10; p /= p.sum(1, keepdims=True)
        return (-p * np.log2(p)).sum(1)

    def _peak_bin(m):  return m.argmax(1).astype(np.float32)
    def _conc(m):      return m.max(1) / (m.sum(1) + 1e-10)
    for pfx, rev in [('PSIZE', False), ('PSIZE_REV', True),
                      ('IPT', False),   ('IPT_REV', True)]:
        cols = [c for c in X.columns
                if pfx.replace('_REV','') in c
                and (('REV' in c) == rev)
                and 'BIN' in c]
        # more specific matching
        if pfx == 'PSIZE':
            cols = [c for c in X.columns if 'PSIZE_BIN' in c and 'REV' not in c]
        elif pfx == 'PSIZE_REV':
            cols = [c for c in X.columns if 'PSIZE_BIN' in c and 'REV' in c]
        elif pfx == 'IPT':
            cols = [c for c in X.columns if 'IPT_BIN' in c and 'REV' not in c]
        elif pfx == 'IPT_REV':
            cols = [c for c in X.columns if 'IPT_BIN' in c and 'REV' in c]
        if len(cols) < 2: continue
        m = X[cols].values.astype(np.float64)
        X[f'{pfx}_ENTROPY']  = _bin_entropy(m)
        X[f'{pfx}_PEAK_BIN'] = _peak_bin(m)
        X[f'{pfx}_CONC']     = _conc(m)
        if pfx in ('PSIZE', 'IPT'):
            X[f'{pfx}_BIN_SUM'] = m.sum(1)
        else:
            X[f'{pfx}_BIN_SUM'] = m.sum(1)
    def log1p_num(df):
        df  = df.copy()
        num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        df[num] = np.log1p(np.maximum(df[num].values, 0.0))
        return df
    X = log1p_num(X)
    return X

def construct_row_from_pcap_file(pcap_file: UploadFile):
    """
    Extracts QUIC flow features from a PCAP file.
    Uses FIRST 30 packets (PPI model assumption).
    Returns a DICT aligned with FEATURE_NAMES BEFORE featurify_157().
    """

    # ---------------------------------------------------------
    # 1. Save uploaded file temporarily
    # ---------------------------------------------------------
    print(f"[INFO] Saving uploaded PCAP file: {pcap_file.filename}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
    tmp.write(pcap_file.file.read())
    tmp.close()
    pcap_path = tmp.name

    # ---------------------------------------------------------
    # 2. Load only QUIC packets
    # ---------------------------------------------------------
    print(f"[INFO] Loading QUIC packets from PCAP: {pcap_path}")
    cap = pyshark.FileCapture(
        pcap_path,
        display_filter="quic"
    )

    packets = []

    try:
        for pkt in cap:
            try:
                ts = float(pkt.sniff_timestamp)
                size = int(pkt.length)

                # UDP fallback safety (QUIC is UDP-based)
                src = getattr(pkt.ip, "src", None)
                dst = getattr(pkt.ip, "dst", None)

                packets.append({
                    "time": ts,
                    "size": size,
                    "src": src,
                    "dst": dst
                })

                if len(packets) >= 30:
                    break

            except Exception:
                continue
    finally:
        cap.close()

    # clean temp file
    os.remove(pcap_path)

    if len(packets) == 0:
        raise ValueError("No QUIC packets found in PCAP")

    print(f"[INFO] Total QUIC packets extracted: {len(packets)}")
    # ---------------------------------------------------------
    # 3. Define client (first packet sender)
    # ---------------------------------------------------------
    client_ip = packets[0]["src"]
    print(f"[INFO] First packet src: {client_ip}, dst: {packets[0]['dst']}")

    # ---------------------------------------------------------
    # 4. Keep only PPI packets
    # ---------------------------------------------------------
    print(f"[INFO] Keeping first 30 packets for PPI features")
    packets = packets[:30]
    ppi_len = len(packets)

    # ---------------------------------------------------------
    # 5. Compute DIR, IPT, SIZE
    # ---------------------------------------------------------
    print(f"[INFO] Computing DIR, IPT, SIZE sequences")
    ipt = []
    sizes = []
    dirs = []

    for i in range(ppi_len):
        pkt = packets[i]

        sizes.append(float(pkt["size"]))

        # direction
        d = 1.0 if pkt["src"] == client_ip else -1.0
        dirs.append(d)

        # IPT
        if i == 0:
            ipt.append(0.0)
        else:
            dt = packets[i]["time"] - packets[i - 1]["time"]
            ipt.append(max(dt * 1000.0, 0.0))  # ms

    # pad to 30
    while len(ipt) < 30:
        ipt.append(0.0)
        sizes.append(0.0)
        dirs.append(0.0)

    # ---------------------------------------------------------
    # 6. Flow statistics
    # ---------------------------------------------------------
    print(f"[INFO] Calculating flow statistics")
    avg_ipt = float(np.mean([x for x in ipt if x > 0]) or 0.0)
    std_ipt = float(np.std(ipt))

    avg_size = float(np.mean(sizes))
    std_size = float(np.std(sizes))

    up = sum(1 for d in dirs if d > 0)
    down = sum(1 for d in dirs if d < 0)

    total_bytes = float(sum(sizes))
    up_bytes = float(sum(s for s, d in zip(sizes, dirs) if d > 0))
    down_bytes = float(sum(s for s, d in zip(sizes, dirs) if d < 0))

    up_packets = up
    down_packets = down

    ul_dl_ratio_direction = up / max(up + down, 1)
    ul_dl_ratio_bytes = up_bytes / max(total_bytes, 1e-6)
    ul_dl_ratio_packets = up_packets / max(up_packets + down_packets, 1)

    # ---------------------------------------------------------
    # 7. PPI metadata
    # ---------------------------------------------------------
    print(f"[INFO] Calculating PPI metadata")
    ppi_roundtrips = 0
    for i in range(1, ppi_len):
        if dirs[i - 1] == 1 and dirs[i] == -1:
            ppi_roundtrips += 1

    ppi_duration = (packets[ppi_len - 1]["time"] - packets[0]["time"]) * 1000.0

    # QUIC PCAP cannot reliably give this → fixed
    flow_end_reason = 2  # OTHER

    # ---------------------------------------------------------
    # 8. Packet size histogram (8 bins)
    # ---------------------------------------------------------
    print(f"[INFO] Calculating packet size histogram")
    def make_hist(values):
        hist, _ = np.histogram(values, bins=8, range=(0, 1500))
        hist = hist.astype(float)
        return hist / (hist.sum() + 1e-6)

    psize_bins = make_hist(sizes)
    ipt_bins = make_hist(ipt)

    # reverse (same distribution mirrored)
    psize_bins_rev = psize_bins[::-1]
    ipt_bins_rev = ipt_bins[::-1]

    # ---------------------------------------------------------
    # 9. Build FULL base feature row (133 features)
    # ---------------------------------------------------------
    print(f"[INFO] Building feature row")
    row = {}

    # IPT / DIR / SIZE sequences
    for i in range(30):
        row[f"IPT_{i+1}"] = float(ipt[i])
        row[f"DIR_{i+1}"] = float(dirs[i])
        row[f"SIZE_{i+1}"] = float(sizes[i])

    # traffic totals
    row["BYTES"] = float(up_bytes)
    row["BYTES_REV"] = float(down_bytes)
    row["PACKETS"] = float(up_packets)
    row["PACKETS_REV"] = float(down_packets)

    row["DURATION"] = float(ppi_duration)
    row["PPI_LEN"] = float(ppi_len)
    row["PPI_ROUNDTRIPS"] = float(ppi_roundtrips)
    row["PPI_DURATION"] = float(ppi_duration)

    # flow end reason
    row["FLOW_ENDREASON_IDLE"] = 1.0 if flow_end_reason == 0 else 0.0
    row["FLOW_ENDREASON_ACTIVE"] = 1.0 if flow_end_reason == 1 else 0.0
    row["FLOW_ENDREASON_OTHER"] = 1.0 if flow_end_reason == 2 else 0.0

    # histogram features
    for i in range(8):
        row[f"PSIZE_BIN{i+1}"] = float(psize_bins[i])
        row[f"IPT_BIN{i+1}"] = float(ipt_bins[i])

        row[f"PSIZE_BIN{i+1}_REV"] = float(psize_bins_rev[i])
        row[f"IPT_BIN{i+1}_REV"] = float(ipt_bins_rev[i])
    print(f"[INFO] Constructed feature row: {row}")
    return row    
# ======================================================================================
# PREDICTION
# ======================================================================================
def predict_pcap(
        pcap_file: UploadFile,
        strategy: str = "avg"
):
    print(f"[INFO] Starting prediction for uploaded PCAP file: {pcap_file.filename}")
    row = construct_row_from_pcap_file(pcap_file)
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    X = featurify_157(X)
    X_scaled = pd.DataFrame(
        SCALER.transform(X),
        columns=FEATURE_NAMES
    )
    probs = []

    start = time.perf_counter()
    for model_name, model in MODELS.items():
        if model_name in TREE_MODELS:
            model_probs = model.predict_proba(X)[0]
            probs.append(model_probs)
        else:
            model_probs = model.predict_proba(X_scaled)[0]
            probs.append(model_probs)
    latency = (time.perf_counter() - start) * 1000
    probs = np.array(probs)
    weighted_probs = np.zeros_like(probs[0])
    weights = get_weights(strategy)
    print(f"[INFO] Using weights for strategy '{strategy}': {weights}")
    for i, weight in enumerate(weights):
        weighted_probs += probs[i] * weight
    pred_idx = int(np.argmax(weighted_probs))
    predicted_label_encoded = int(
        MODELS[list(MODELS.keys())[0]].classes_[pred_idx]
    )
    application_name = APP_ID_TO_NAME.get(
        predicted_label_encoded,
        f"unknown-{predicted_label_encoded}"
    )
    confidence = float(weighted_probs[pred_idx])
    top5_indices = np.argsort(weighted_probs)[::-1][:5]
    top5_predictions = []
    for idx in top5_indices:
        app_id = int(
            MODELS[list(MODELS.keys())[0]].classes_[idx]
        )
        top5_predictions.append({
            "app_id": app_id,
            "app_name": APP_ID_TO_NAME.get(app_id, "unknown"),
            "probability": round(float(weighted_probs[idx]), 6),
        })
    return {
        "strategy": strategy,
        "prediction": {
            "app_id": predicted_label_encoded,
            "app_name": application_name,
            "confidence": round(confidence, 6),
        },
        "top5_predictions": top5_predictions,
        "latency_ms": round(latency, 3),
    }
def predict(features: Dict[str, float], strategy: str = "avg") -> Dict:


    row = construct_row_v2(features)
    print_construct_row_v2_stats(row)
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    X = featurify_157(X)
    # for c in X.columns:
    #     print(f"{X[c]} -> {_X[c]}")
    # X = _X
    X_scaled = pd.DataFrame(
        SCALER.transform(X),
        columns=FEATURE_NAMES
    )
    probs = []

    start = time.perf_counter()
    for model_name, model in MODELS.items():
        # if model is a tree then fit without scaling
        if model_name in TREE_MODELS:
            model_probs = model.predict_proba(X)[0]
            probs.append(model_probs)
        else:
            model_probs = model.predict_proba(X_scaled)[0]
            probs.append(model_probs)
    latency = (time.perf_counter() - start) * 1000

    probs = np.array(probs)

    weighted_probs = np.zeros_like(probs[0])
    weights = get_weights(strategy)
    print(f"[INFO] Using weights for strategy '{strategy}': {weights}")
    for i, weight in enumerate(weights):

        weighted_probs += probs[i] * weight

    pred_idx = int(np.argmax(weighted_probs))

    predicted_label_encoded = int(
        MODELS[list(MODELS.keys())[0]].classes_[pred_idx]
    )

    application_name = APP_ID_TO_NAME.get(
        predicted_label_encoded,
        f"unknown-{predicted_label_encoded}"
    )

    confidence = float(weighted_probs[pred_idx])
    total_conf = np.sum(weighted_probs)
    print("Total conf:", total_conf)
    top5_indices = np.argsort(weighted_probs)[::-1][:5]

    top5_predictions = []

    for idx in top5_indices:

        app_id = int(
            MODELS[list(MODELS.keys())[0]].classes_[idx]
        )

        top5_predictions.append({
            "app_id": app_id,
            "app_name": APP_ID_TO_NAME.get(app_id, "unknown"),
            "probability": round(float(weighted_probs[idx]), 6),
        })

    return {
        "strategy": strategy,
        "prediction": {
            "app_id": predicted_label_encoded,
            "app_name": application_name,
            "confidence": round(confidence, 6),
        },
        "top5_predictions": top5_predictions,
        "latency_ms": round(latency, 3),
    }
