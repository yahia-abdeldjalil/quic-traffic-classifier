"""
quic-backend/app/pcap_processor.py — Extract QUIC flow features directly from a .pcap / .pcapng file.

Builds the 133-feature base row (IPT_1..30, SIZE_1..30, DIR_1..30, histogram bins, totals)
from real per-packet data.  featurify_157 in predictor.py handles the 24 derived features.

Usage:
    base_features, summary, meta = extract_pcap_features(pcap_bytes)
"""

import io
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppress scapy verbose output
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
logging.getLogger("scapy.loading").setLevel(logging.ERROR)

try:
    from scapy.all import IP, IPv6, UDP, PcapReader
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False

# ─── constants ───────────────────────────────────────────────────────────────
QUIC_PORTS    = {443, 80, 8443, 8080, 4433, 9000}
MAX_PKT_READ  = 5_000   # cap file read for large captures
PPI_WINDOW    = 30      # first N packets per flow (same as training)

PSIZE_EDGES = [0, 128, 256, 384, 512, 640, 768, 1024, float("inf")]
IPT_EDGES   = [0, 1,   10,  50, 100, 200, 500, 1000, float("inf")]


# ─── helpers ─────────────────────────────────────────────────────────────────
def _hist_norm(values: np.ndarray, edges: list) -> np.ndarray:
    if len(values) == 0:
        return np.ones(8) / 8.0
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return counts.astype(float) / total if total > 0 else np.ones(8) / 8.0


def _flow_key_and_direction(src_ip, sport, dst_ip, dport):
    """
    Return (canonical_key, direction).
    canonical_key = (client_ip, client_port, server_ip, server_port)
    direction = +1.0 if this packet is client→server (upload), -1.0 otherwise.
    """
    dst_is_server = dport in QUIC_PORTS
    src_is_server = sport in QUIC_PORTS

    if dst_is_server and not src_is_server:
        return (src_ip, sport, dst_ip, dport), 1.0   # client → server
    if src_is_server and not dst_is_server:
        return (dst_ip, dport, src_ip, sport), -1.0  # server → client

    # Neither port is a known QUIC port — use lower port as server
    if dport <= sport:
        return (src_ip, sport, dst_ip, dport), 1.0
    return (dst_ip, dport, src_ip, sport), -1.0


# ─── main public function ─────────────────────────────────────────────────────
def extract_pcap_features(
    pcap_bytes: bytes,
) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
    """
    Parse a raw PCAP / PCAPng byte blob and extract QUIC flow features.

    Returns
    -------
    base_features : dict  — 133 named features (keys match notebook FEATURE_NAMES[0:133])
    summary_stats : dict  — high-level stats for frontend display (avg_ipt, etc.)
    meta          : dict  — PCAP metadata (n_flows, selected flow, etc.)
    """
    if not SCAPY_OK:
        raise RuntimeError(
            "scapy is required for PCAP processing.  Install with:  pip install scapy"
        )

    # write to temp file (scapy needs a path, not a buffer)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(pcap_bytes)
        tmp_path = tmp.name

    try:
        return _process(tmp_path)
    finally:
        os.unlink(tmp_path)


# ─── internal processing ─────────────────────────────────────────────────────
def _process(path: str):
    # ── 1. read packets (capped) ──────────────────────────────────────────────
    packets_raw = []
    with PcapReader(path) as reader:
        for i, pkt in enumerate(reader):
            if i >= MAX_PKT_READ:
                break
            packets_raw.append(pkt)

    if not packets_raw:
        raise ValueError("PCAP file is empty or unreadable.")

    # ── 2. group into flows ───────────────────────────────────────────────────
    flows: Dict[tuple, List[Tuple[float, int, float]]] = {}

    for pkt in packets_raw:
        has_ip = IP in pkt
        has_ip6 = IPv6 in pkt
        if not (has_ip or has_ip6):
            continue
        if UDP not in pkt:
            continue

        src_ip = pkt[IP].src   if has_ip  else pkt[IPv6].src
        dst_ip = pkt[IP].dst   if has_ip  else pkt[IPv6].dst
        sport  = pkt[UDP].sport
        dport  = pkt[UDP].dport
        ts     = float(pkt.time)
        size   = int(pkt[IP].len if has_ip else len(pkt))  # IP datagram size

        key, direction = _flow_key_and_direction(src_ip, sport, dst_ip, dport)
        flows.setdefault(key, []).append((ts, size, direction))

    if not flows:
        raise ValueError(
            "No UDP flows found in this PCAP.  "
            "Please capture QUIC/UDP traffic (typically on port 443)."
        )

    # ── 3. select best flow ───────────────────────────────────────────────────
    def _score(item):
        key, pkts = item
        _c_ip, _c_p, _s_ip, s_port = key
        is_quic = int(s_port in QUIC_PORTS or _c_p in QUIC_PORTS)
        return is_quic * 1_000_000 + len(pkts)

    best_key, best_pkts = max(flows.items(), key=_score)
    best_pkts.sort(key=lambda x: x[0])   # sort by timestamp

    n_flows        = len(flows)
    c_ip, c_port, s_ip, s_port = best_key
    flow_label     = f"{c_ip}:{c_port} → {s_ip}:{s_port}"
    total_in_flow  = len(best_pkts)
    is_quic        = s_port in QUIC_PORTS or c_port in QUIC_PORTS

    # ── 4. PPI window (first 30 packets) ─────────────────────────────────────
    ppi_len  = min(PPI_WINDOW, len(best_pkts))
    ppi      = best_pkts[:ppi_len]

    ts_arr   = np.array([p[0] for p in ppi], dtype=np.float64)
    sz_arr   = np.array([p[1] for p in ppi], dtype=np.float64)
    dir_arr  = np.array([p[2] for p in ppi], dtype=np.float64)

    # IPT (ms): first packet gets IPT=0, subsequent get delta from previous
    if ppi_len > 1:
        ipt_arr = np.concatenate([[0.0], np.diff(ts_arr) * 1000.0])
    else:
        ipt_arr = np.zeros(1)
    ipt_arr = np.maximum(0.0, ipt_arr)

    # ── 5. per-packet base features (IPT_i, SIZE_i, DIR_i) ──────────────────
    base: Dict[str, float] = {}
    for i in range(30):
        if i < ppi_len:
            base[f"IPT_{i+1}"]  = float(ipt_arr[i])
            base[f"SIZE_{i+1}"] = float(max(1.0, sz_arr[i]))
            base[f"DIR_{i+1}"]  = float(dir_arr[i])
        else:                            # zero-pad unused slots
            base[f"IPT_{i+1}"]  = 0.0
            base[f"SIZE_{i+1}"] = 0.0
            base[f"DIR_{i+1}"]  = 0.0

    # ── 6. traffic totals ────────────────────────────────────────────────────
    up_mask   = dir_arr > 0
    down_mask = dir_arr < 0
    bytes_up   = float(sz_arr[up_mask].sum())   if up_mask.any()   else 0.0
    bytes_down = float(sz_arr[down_mask].sum()) if down_mask.any() else 0.0

    base["BYTES"]       = bytes_up
    base["BYTES_REV"]   = bytes_down
    base["PACKETS"]     = float(up_mask.sum())
    base["PACKETS_REV"] = float(down_mask.sum())

    # ── 7. metadata ──────────────────────────────────────────────────────────
    ppi_duration_s = float(ts_arr[-1] - ts_arr[0]) if ppi_len > 1 else 0.0
    dir_changes    = int(np.sum(np.abs(np.diff(dir_arr)) > 0)) if ppi_len > 1 else 0
    roundtrips     = float(max(0, dir_changes // 2))

    base["DURATION"]       = ppi_duration_s
    base["PPI_LEN"]        = float(ppi_len)
    base["PPI_ROUNDTRIPS"] = roundtrips
    base["PPI_DURATION"]   = ppi_duration_s

    base["FLOW_ENDREASON_IDLE"]   = 1.0
    base["FLOW_ENDREASON_ACTIVE"] = 0.0
    base["FLOW_ENDREASON_OTHER"]  = 0.0

    # ── 8. histogram bins (from real data) ───────────────────────────────────
    up_sizes   = sz_arr[up_mask]   if up_mask.any()   else sz_arr
    down_sizes = sz_arr[down_mask] if down_mask.any() else sz_arr
    up_ipts    = ipt_arr[up_mask[:len(ipt_arr)]]   if up_mask.any()   else ipt_arr
    down_ipts  = ipt_arr[down_mask[:len(ipt_arr)]] if down_mask.any() else ipt_arr

    psize_fwd = _hist_norm(up_sizes,   PSIZE_EDGES)
    psize_rev = _hist_norm(down_sizes, PSIZE_EDGES)
    ipt_fwd   = _hist_norm(up_ipts,    IPT_EDGES)
    ipt_rev   = _hist_norm(down_ipts,  IPT_EDGES)

    for i in range(8):
        base[f"PSIZE_BIN{i+1}"]     = float(psize_fwd[i])
        base[f"PSIZE_BIN{i+1}_REV"] = float(psize_rev[i])
        base[f"IPT_BIN{i+1}"]       = float(ipt_fwd[i])
        base[f"IPT_BIN{i+1}_REV"]   = float(ipt_rev[i])

    # ── 9. summary stats (for frontend display, matching slider labels) ───────
    active_ipts  = ipt_arr[1:] if ppi_len > 1 else ipt_arr  # skip the 0 at index 0
    summary = {
        "avg_ipt":               round(float(active_ipts.mean()) if len(active_ipts) else 0.0, 2),
        "std_ipt":               round(float(active_ipts.std(ddof=0)) if len(active_ipts) else 0.0, 2),
        "avg_size":              round(float(sz_arr.mean()), 2),
        "std_size":              round(float(sz_arr.std(ddof=0)), 2),
        "ul_dl_ratio_direction": round(float(np.clip(up_mask.mean(), 0, 1)), 3),
        "ul_dl_ratio_bytes":     round(float(np.clip(bytes_up / max(bytes_up + bytes_down, 1e-9), 0, 1)), 3),
        "ul_dl_ratio_packets":   round(float(np.clip(up_mask.mean(), 0, 1)), 3),
        "ppi_len":               float(ppi_len),
        "ppi_roundtrips":        roundtrips,
        "ppi_duration":          round(ppi_duration_s * 1000, 1),   # in ms for display
        "flow_end_reason":       0.0,
    }

    meta = {
        "n_flows_detected":  n_flows,
        "selected_flow":     flow_label,
        "server_port":       int(s_port),
        "is_quic_port":      is_quic,
        "total_pkts_in_flow": total_in_flow,
        "ppi_pkts_used":     ppi_len,
        "truncated_at":      MAX_PKT_READ,
    }

    return base, summary, meta