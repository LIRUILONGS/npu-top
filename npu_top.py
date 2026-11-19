#!/usr/bin/env python3
"""npu-top: a tiny zero-dependency terminal dashboard for vLLM + Ascend NPU.

The tool reads two read-only sources:
  * a vLLM Prometheus endpoint (default: http://127.0.0.1:8000/metrics)
  * ``npu-smi info`` for Ascend hardware counters

It intentionally has no third-party dependencies and does not write any state.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

__version__ = "0.1.0"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000"
MAX_METRICS_BYTES = 16 * 1024 * 1024

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

SPARK = "▁▂▃▄▅▆▇█"


STRINGS = {
    "en": {
        "subtitle": "vLLM + Ascend NPU",
        "concurrency": "Concurrency",
        "waiting": "waiting",
        "swap": "swap",
        "kv": "KV cache",
        "throughput": "Throughput",
        "prompt": "prompt",
        "generation": "generation",
        "requests": "requests",
        "latency": "Latency",
        "queue": "queue",
        "spec": "Spec decode",
        "accept": "accept",
        "preempt": "preempt",
        "success": "success",
        "total": "total",
        "in": "in",
        "out": "out",
        "trend": "trend",
        "running": "running",
        "throughput_chart_title": "Throughput bar chart",
        "throughput_chart_scale": "last 30 samples · independent scales · tok/s",
        "peak": "peak",
        "avg": "avg",
        "chart_oldest": "oldest",
        "chart_now": "now",
        "npu_title": "NPU AICore / HBM / Temp / Power",
        "aicore": "AICore",
        "hbm": "HBM",
        "temp": "Temp",
        "power": "Power",
        "avg_aicore": "avg AICore",
        "max_temp": "max temp",
        "total_power": "total power",
        "offline": "offline",
        "no_npu": "No NPU data",
        "chart_title": "NPU chart",
        "chart_legend": "AICore / HBM / Temp / Power",
        "chart_scale": "scale: AICore 0-100%, HBM 0-100%, Temp 0-100C, Power 0-300W",
        "quit_hint": "q quit  ·  Ctrl-C stop  ·  refresh {interval:.1f}s",
        "endpoint": "endpoint",
    },
    "zh": {
        "subtitle": "vLLM + 昇腾 NPU",
        "concurrency": "并发",
        "waiting": "排队",
        "swap": "换出",
        "kv": "KV缓存",
        "throughput": "吞吐",
        "prompt": "输入",
        "generation": "生成",
        "requests": "请求",
        "latency": "延迟",
        "queue": "队列",
        "spec": "投机解码",
        "accept": "接受",
        "preempt": "抢占",
        "success": "成功",
        "total": "累计",
        "in": "入",
        "out": "出",
        "trend": "趋势",
        "running": "运行",
        "throughput_chart_title": "吞吐柱状图",
        "throughput_chart_scale": "最近 30 个采样点 · 独立量程 · tok/s",
        "peak": "峰值",
        "avg": "平均",
        "chart_oldest": "最早",
        "chart_now": "现在",
        "npu_title": "NPU 算力 / 显存 / 温度 / 功耗",
        "aicore": "算力",
        "hbm": "显存",
        "temp": "温度",
        "power": "功耗",
        "avg_aicore": "平均算力",
        "max_temp": "最高温度",
        "total_power": "总功耗",
        "offline": "离线",
        "no_npu": "无 NPU 数据",
        "chart_title": "NPU 柱状图",
        "chart_legend": "算力 / 显存 / 温度 / 功耗",
        "chart_scale": "量程：算力 0-100%，显存 0-100%，温度 0-100℃，功耗 0-300W",
        "quit_hint": "q 退出  ·  Ctrl-C 停止  ·  刷新 {interval:.1f}s",
        "endpoint": "端点",
    },
}


class Translator:
    def __init__(self, lang: str) -> None:
        self.lang = lang if lang in STRINGS else "en"
        self.table = STRINGS[self.lang]

    def __call__(self, key: str, **kwargs: object) -> str:
        text = self.table.get(key, key)
        return text.format(**kwargs) if kwargs else text


def resolve_lang(value: str) -> str:
    if value and value != "auto":
        return value if value in STRINGS else "en"
    locale = (os.environ.get("LC_ALL") or os.environ.get("LANG") or "").lower()
    return "zh" if locale.startswith("zh") else "en"


@dataclass(frozen=True)
class Theme:
    enabled: bool = True

    def paint(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.enabled else text

    def bold(self, text: str) -> str:
        return self.paint(text, BOLD)

    def dim(self, text: str) -> str:
        return self.paint(text, DIM)

    def red(self, text: str) -> str:
        return self.paint(text, RED)

    def green(self, text: str) -> str:
        return self.paint(text, GREEN)

    def yellow(self, text: str) -> str:
        return self.paint(text, YELLOW)


@dataclass
class Endpoint:
    name: str
    url: str
    api_key: str = ""
    max_seqs: int = 0


@dataclass
class ServiceSnapshot:
    name: str
    url: str
    online: bool = False
    error: str = ""
    model: str = ""
    max_seqs: int = 0
    running: int = 0
    waiting: int = 0
    swapped: int = 0
    kv: float = 0.0
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    req_tps: float = 0.0
    ttft: float = 0.0
    tpot: float = 0.0
    e2e: float = 0.0
    queue: float = 0.0
    prefill: float = 0.0
    decode: float = 0.0
    itl: float = 0.0
    preempt: int = 0
    mtp: float = 0.0
    success_total: float = 0.0
    prompt_total: float = 0.0
    gen_total: float = 0.0
    history: List[Tuple[int, float]] = field(default_factory=list)
    throughput_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class NpuSnapshot:
    id: int
    name: str
    health: str
    power: float
    temp: int
    aicore: int
    hbm_used: int
    hbm_total: int
    hbm_pct: float


SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)(?:\s+\S+)?$"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')

HISTOGRAMS = (
    ("vllm:time_to_first_token_seconds", "ttft"),
    ("vllm:e2e_request_latency_seconds", "e2e"),
    ("vllm:request_queue_time_seconds", "queue"),
    ("vllm:request_prefill_time_seconds", "prefill"),
    ("vllm:request_decode_time_seconds", "decode"),
    ("vllm:request_time_per_output_token_seconds", "tpot"),
    ("vllm:inter_token_latency_seconds", "itl"),
)


def _unescape_label(value: str) -> str:
    return value.replace('\\"', '"').replace("\\\\", "\\")


def parse_prometheus(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    samples: Dict[str, List[Tuple[Dict[str, str], float]]] = collections.defaultdict(list)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        labels = {
            key: _unescape_label(value)
            for key, value in LABEL_RE.findall(match.group(2) or "")
        }
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        samples[match.group(1)].append((labels, value))
    return dict(samples)


def metric_sum(
    samples: Dict[str, List[Tuple[Dict[str, str], float]]], *names: str
) -> float:
    for name in names:
        points = samples.get(name)
        if points:
            return sum(value for _labels, value in points)
    return 0.0


def metric_max(
    samples: Dict[str, List[Tuple[Dict[str, str], float]]], *names: str
) -> float:
    for name in names:
        points = samples.get(name)
        if points:
            return max(value for _labels, value in points)
    return 0.0


def metric_model(samples: Dict[str, List[Tuple[Dict[str, str], float]]]) -> str:
    for points in samples.values():
        for labels, _value in points:
            for key in ("model_name", "model"):
                if labels.get(key):
                    return labels[key]
    return ""


def normalize_metrics_url(url: str) -> str:
    value = url.strip()
    if "://" not in value:
        value = "http://" + value
    parts = urllib.parse.urlsplit(value)
    if not parts.netloc:
        raise ValueError(f"invalid endpoint: {url!r}")
    path = parts.path
    if path in ("", "/"):
        path = "/metrics"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def label_from_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or parts.netloc or "endpoint"
    return f"{host}:{parts.port}" if parts.port else host


def parse_endpoint_spec(spec: str) -> Tuple[str, str]:
    value = spec.strip()
    if not value:
        raise ValueError("empty endpoint")
    if "=" in value and not value.startswith(("http://", "https://")):
        name, raw_url = value.split("=", 1)
    else:
        name, raw_url = "", value
    url = normalize_metrics_url(raw_url)
    name = name.strip() or label_from_url(url)
    return name, url


def fetch_metrics(endpoint: Endpoint, timeout: float) -> str:
    headers = {
        "Accept": "text/plain",
        "User-Agent": f"npu-top/{__version__}",
    }
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    request = urllib.request.Request(endpoint.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"HTTP {status}")
        payload = response.read(MAX_METRICS_BYTES + 1)
    if len(payload) > MAX_METRICS_BYTES:
        raise RuntimeError("metrics response is larger than 16 MiB")
    return payload.decode("utf-8", "ignore")


def short_error(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason}".strip()
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)[:160]
    if isinstance(exc, TimeoutError):
        return "request timed out"
    return str(exc)[:160]


def _delta(current: float, previous: float) -> float:
    return max(0.0, current - previous)


def build_snapshot(
    endpoint: Endpoint,
    text: str,
    previous: Optional[Dict[str, float]],
    dt: float,
    average_state: Dict[str, float],
) -> Tuple[ServiceSnapshot, Dict[str, float]]:
    samples = parse_prometheus(text)
    flat = {
        name: sum(value for _labels, value in points)
        for name, points in samples.items()
    }
    snapshot = ServiceSnapshot(
        name=endpoint.name,
        url=endpoint.url,
        online=True,
        model=metric_model(samples),
        max_seqs=endpoint.max_seqs,
        running=int(metric_sum(samples, "vllm:num_requests_running")),
        waiting=int(metric_sum(samples, "vllm:num_requests_waiting")),
        swapped=int(
            metric_sum(samples, "vllm:num_requests_swapped")
        ),
        success_total=metric_sum(samples, "vllm:request_success_total"),
        prompt_total=metric_sum(samples, "vllm:prompt_tokens_total"),
        gen_total=metric_sum(samples, "vllm:generation_tokens_total"),
    )

    kv = metric_max(
        samples, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"
    )
    if kv > 1.5:
        kv /= 100.0
    snapshot.kv = min(1.0, max(0.0, kv))

    if previous:
        span = max(dt, 0.001)
        snapshot.prompt_tps = _delta(
            snapshot.prompt_total, previous.get("vllm:prompt_tokens_total", 0.0)
        ) / span
        snapshot.gen_tps = _delta(
            snapshot.gen_total, previous.get("vllm:generation_tokens_total", 0.0)
        ) / span
        snapshot.req_tps = _delta(
            snapshot.success_total, previous.get("vllm:request_success_total", 0.0)
        ) / span
        snapshot.preempt = int(
            _delta(
                flat.get("vllm:num_preemptions_total", 0.0),
                previous.get("vllm:num_preemptions_total", 0.0),
            )
        )
        drafts = _delta(
            flat.get("vllm:spec_decode_num_draft_tokens_total", 0.0),
            previous.get("vllm:spec_decode_num_draft_tokens_total", 0.0),
        )
        accepted = _delta(
            flat.get("vllm:spec_decode_num_accepted_tokens_total", 0.0),
            previous.get("vllm:spec_decode_num_accepted_tokens_total", 0.0),
        )
        if drafts > 0:
            snapshot.mtp = accepted * 100.0 / drafts

    if snapshot.mtp <= 0:
        drafts = flat.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
        accepted = flat.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
        if drafts > 0:
            snapshot.mtp = accepted * 100.0 / drafts

    for base, field_name in HISTOGRAMS:
        count = flat.get(base + "_count", 0.0)
        total = flat.get(base + "_sum", 0.0)
        value = average_state.get(base, 0.0)
        if count > 0:
            if previous:
                delta_count = count - previous.get(base + "_count", 0.0)
                delta_sum = total - previous.get(base + "_sum", 0.0)
                if delta_count > 0 and delta_sum >= 0:
                    value = delta_sum / delta_count
                elif count < previous.get(base + "_count", 0.0):
                    value = total / count
                else:
                    value = average_state.get(base, total / count)
            else:
                value = total / count
        if value > 0:
            average_state[base] = value
        setattr(snapshot, field_name, value)

    return snapshot, flat


class Monitor:
    def __init__(
        self,
        endpoints: Sequence[Endpoint],
        npu_binary: Optional[str],
        timeout: float,
        npu_enabled: bool = True,
    ) -> None:
        self.endpoints = list(endpoints)
        self.npu_binary = npu_binary
        self.timeout = timeout
        self.npu_enabled = npu_enabled
        self.previous: Dict[str, Dict[str, float]] = {}
        self.average: Dict[str, Dict[str, float]] = {
            endpoint.name: {} for endpoint in self.endpoints
        }
        self.history: Dict[str, Deque[Tuple[int, float]]] = {
            endpoint.name: collections.deque(maxlen=60) for endpoint in self.endpoints
        }
        self.throughput_history: Dict[str, Deque[Tuple[float, float]]] = {
            endpoint.name: collections.deque(maxlen=60) for endpoint in self.endpoints
        }
        self.npu_history: Dict[int, Deque[Tuple[int, float]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=60)
        )
        self.last_time = time.monotonic()

    def poll(self) -> Tuple[List[ServiceSnapshot], List[NpuSnapshot], str]:
        now = time.monotonic()
        dt = now - self.last_time
        self.last_time = now
        services = [self._poll_service(endpoint, dt) for endpoint in self.endpoints]
        if self.npu_enabled:
            npus, npu_error = read_npus(self.npu_binary, timeout=8.0)
        else:
            npus, npu_error = [], ""
        for npu in npus:
            self.npu_history[npu.id].append((npu.aicore, npu.hbm_pct))
        return services, npus, npu_error

    def _poll_service(self, endpoint: Endpoint, dt: float) -> ServiceSnapshot:
        try:
            text = fetch_metrics(endpoint, self.timeout)
            snapshot, flat = build_snapshot(
                endpoint,
                text,
                self.previous.get(endpoint.name),
                dt,
                self.average[endpoint.name],
            )
            self.previous[endpoint.name] = flat
        except Exception as exc:
            self.previous.pop(endpoint.name, None)
            snapshot = ServiceSnapshot(
                name=endpoint.name,
                url=endpoint.url,
                online=False,
                error=short_error(exc),
                max_seqs=endpoint.max_seqs,
            )
        history = self.history[endpoint.name]
        history.append((snapshot.running, snapshot.gen_tps))
        snapshot.history = list(history)
        throughput = self.throughput_history[endpoint.name]
        throughput.append((snapshot.prompt_tps, snapshot.gen_tps))
        snapshot.throughput_history = list(throughput)
        return snapshot


NPU_HEAD_RE = re.compile(
    r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(\S+)\s*\|\s*([0-9.]+|-|N/A)\s+(-?\d+|-)\s+",
    re.IGNORECASE,
)
NPU_CHIP_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*(\d+)\s+(.*)\|\s*$"
)


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_npu_info(text: str) -> List[NpuSnapshot]:
    found: Dict[int, NpuSnapshot] = {}
    pending: Optional[NpuSnapshot] = None
    for line in text.splitlines():
        head = NPU_HEAD_RE.match(line)
        if head:
            pending = NpuSnapshot(
                id=_to_int(head.group(1)),
                name=head.group(2),
                health=head.group(3),
                power=_to_float(head.group(4)),
                temp=_to_int(head.group(5)),
                aicore=0,
                hbm_used=0,
                hbm_total=0,
                hbm_pct=0.0,
            )
            continue
        chip = NPU_CHIP_RE.match(line)
        if chip and pending is not None:
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", chip.group(2))
            if not pairs:
                continue
            used, total = (int(part) for part in pairs[-1])
            pending.aicore = _to_int(chip.group(1))
            pending.hbm_used = used
            pending.hbm_total = total
            pending.hbm_pct = used * 100.0 / total if total else 0.0
            found.setdefault(pending.id, pending)
            pending = None
    return [found[key] for key in sorted(found)]


def read_npus(binary: Optional[str], timeout: float = 8.0) -> Tuple[List[NpuSnapshot], str]:
    executable = shutil.which(binary) if binary else shutil.which("npu-smi")
    if not executable:
        return [], "npu-smi not found (install the Ascend driver tools or use --no-npu)"
    try:
        process = subprocess.run(
            [executable, "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], "npu-smi timed out"
    except OSError as exc:
        return [], f"npu-smi failed: {exc}"
    output = process.stdout or ""
    if process.returncode != 0 and not output:
        detail = (process.stderr or "").strip().splitlines()
        return [], f"npu-smi exited {process.returncode}" + (f": {detail[0][:120]}" if detail else "")
    npus = parse_npu_info(output)
    if not npus:
        return [], "npu-smi returned no parsable NPU rows"
    return npus, ""


def bar(ratio: float, width: int = 16) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def ratio_color(ratio: float) -> str:
    if ratio >= 0.9:
        return RED
    if ratio >= 0.7:
        return YELLOW
    return GREEN


def colored_bar(ratio: float, width: int, theme: Theme) -> str:
    return theme.paint(bar(ratio, width), ratio_color(ratio))


def temp_color(value: int) -> str:
    if value >= 75:
        return RED
    if value >= 65:
        return YELLOW
    return GREEN


def power_color(value: float) -> str:
    if value >= 250:
        return RED
    if value >= 200:
        return YELLOW
    return GREEN


def sparkline(values: Sequence[float], width: int = 24, ceiling: Optional[float] = None) -> str:
    points = list(values)[-width:]
    if not points:
        return ""
    top = ceiling if ceiling is not None else max(points)
    top = top or 1.0
    return "".join(
        SPARK[min(len(SPARK) - 1, int(max(0.0, value) / top * (len(SPARK) - 1)))]
        for value in points
    )


def fmt_count(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def fmt_rate(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}K"
    if value >= 1_000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def fmt_sec(value: float) -> str:
    if value <= 0:
        return "-"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    return f"{value:.2f}s"


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _bar_grid(
    values: Sequence[float],
    width: int,
    height: int,
    bar_width: int,
    gap: int,
) -> List[List[str]]:
    top = max(values) if values else 0.0
    top = top or 1.0
    step = bar_width + gap
    max_bars = max(1, (width + gap) // step)
    recent = list(values)[-max_bars:]
    offset = width - (len(recent) * step - gap)
    grid = [[" "] * width for _ in range(height)]

    def filled_rows(value: float) -> int:
        ratio = max(0.0, min(1.0, value / top))
        filled = int(round(ratio * height))
        if value > 0 and filled == 0:
            filled = 1
        return filled

    for index, value in enumerate(recent):
        x = offset + index * step
        filled = filled_rows(value)
        for row in range(height - filled, height):
            for column in range(x, x + bar_width):
                grid[row][column] = "█"
    return grid


def render_throughput_chart(
    history: Sequence[Tuple[float, float]],
    t: Translator,
    theme: Theme,
    width: int,
) -> List[str]:
    if not history:
        return []
    bar_width = 3 if width >= 128 else 2
    gap = 1
    max_bars = max(8, min(30, (width - 14 + gap) // (bar_width + gap)))
    plot_width = max_bars * (bar_width + gap) - gap
    height = 5
    lines = [
        f"{theme.bold(t('throughput_chart_title'))}  "
        f"{theme.dim(t('throughput_chart_scale'))}"
    ]

    def panel(label: str, values: Sequence[float], color: str) -> List[str]:
        top = max(values) if values else 0.0
        top = top or 1.0
        avg = sum(values) / len(values) if values else 0.0
        result = [
            f"{theme.bold(label)}  tok/s  "
            f"{theme.dim(t('peak'))} {fmt_rate(top)}  "
            f"{theme.dim(t('avg'))} {fmt_rate(avg)}"
        ]
        grid = _bar_grid(values, plot_width, height, bar_width, gap)
        for row in range(height):
            if row == 0:
                prefix = f"{fmt_rate(top):>8} ┤"
            elif row == height - 1:
                prefix = f"{0:>8} ┤"
            elif row == height // 2:
                prefix = f"{fmt_rate(top / 2):>8} │"
            else:
                prefix = " " * 8 + " │"
            result.append(theme.dim(prefix) + theme.paint("".join(grid[row]), color))
        result.append(theme.dim(" " * 8 + " └" + "─" * plot_width))
        oldest = t("chart_oldest")
        now = t("chart_now")
        label_gap = max(1, plot_width - len(oldest) - len(now))
        result.append(theme.dim(" " * 10 + oldest + " " * label_gap + now))
        return result

    lines.extend(panel(t("prompt"), [point[0] for point in history], CYAN))
    lines.extend(panel(t("generation"), [point[1] for point in history], GREEN))
    return lines


def render_service(
    snapshot: ServiceSnapshot,
    t: Translator,
    theme: Theme,
    spark: bool = False,
    width: int = 100,
) -> List[str]:
    url = truncate(snapshot.url, max(20, width - len(snapshot.name) - len(snapshot.model) - 8))
    lines = [f"{theme.bold(snapshot.name)}  {snapshot.model or '-'}  {theme.dim(url)}"]
    if not snapshot.online:
        detail = f": {snapshot.error}" if snapshot.error else ""
        lines.append(f"  {theme.red(t('offline'))}{detail}")
        return lines

    if snapshot.max_seqs > 0:
        ratio = snapshot.running / snapshot.max_seqs
        concurrency = (
            f"{colored_bar(ratio, 16, theme)} {snapshot.running}/{snapshot.max_seqs}"
        )
    else:
        concurrency = str(snapshot.running)
    waiting = theme.paint(str(snapshot.waiting), RED if snapshot.waiting else GREEN)
    line = (
        f"  {t('concurrency')}  {concurrency}"
        f"   {t('waiting')} {waiting}"
        f"   {t('kv')} {colored_bar(snapshot.kv, 12, theme)} {snapshot.kv * 100:5.1f}%"
    )
    if snapshot.swapped:
        line += f"   {t('swap')} {snapshot.swapped}"
    lines.append(line)
    lines.append(
        f"  {t('throughput')}  {t('prompt')} {snapshot.prompt_tps:8.1f} tok/s"
        f"   {t('generation')} {snapshot.gen_tps:8.1f} tok/s"
        f"   {t('requests')} {snapshot.req_tps:5.2f}/s"
    )
    lines.append(
        f"  {t('latency')}  TTFT {fmt_sec(snapshot.ttft):>7}"
        f"  TPOT {fmt_sec(snapshot.tpot):>7}"
        f"  E2E {fmt_sec(snapshot.e2e):>7}"
        f"  {t('queue')} {fmt_sec(snapshot.queue):>7}"
    )
    mtp = f"{snapshot.mtp:5.1f}%" if snapshot.mtp > 0 else "-"
    lines.append(
        f"  {t('spec')} {t('accept')} {mtp:>6}"
        f"   {t('preempt')} {snapshot.preempt}"
        f"   {t('success')} {fmt_count(snapshot.success_total)}"
        f"   {t('total')} {t('in')} {fmt_count(snapshot.prompt_total)}"
        f" / {t('out')} {fmt_count(snapshot.gen_total)}"
    )
    if spark and snapshot.history:
        running_curve = sparkline([point[0] for point in snapshot.history], 24)
        generation_curve = sparkline([point[1] for point in snapshot.history], 24)
        trend = (
            t("trend")
            + "  "
            + t("running")
            + " "
            + running_curve
            + "  "
            + t("generation")
            + " "
            + generation_curve
        )
        lines.append(f"  {theme.dim(trend)}")
    return lines


def render_npus(
    npus: Sequence[NpuSnapshot],
    npu_error: str,
    history: Dict[int, Deque[Tuple[int, float]]],
    t: Translator,
    theme: Theme,
    spark: bool = False,
) -> List[str]:
    lines = [theme.bold(t("npu_title"))]
    if not npus:
        lines.append(f"  {theme.red(npu_error or t('no_npu'))}")
        return lines
    for npu in npus:
        temp = theme.paint(f"{npu.temp:2d}C", temp_color(npu.temp))
        power = theme.paint(f"{npu.power:5.1f}W", power_color(npu.power))
        health = npu.health if npu.health == "OK" else theme.red(npu.health)
        extra = ""
        if spark:
            extra = f"  {theme.dim(sparkline([p[0] for p in history[npu.id]], 10, ceiling=100))}"
        lines.append(
            f"  NPU{npu.id:<2} {truncate(npu.name, 9):<9} {health:<5}"
            f"  {t('aicore')} {colored_bar(npu.aicore / 100.0, 10, theme)} {npu.aicore:3d}%"
            f"  {t('hbm')} {colored_bar(npu.hbm_pct / 100.0, 12, theme)} {npu.hbm_pct:5.1f}%"
            f" ({npu.hbm_used / 1024:5.1f}/{npu.hbm_total / 1024:5.1f}G)"
            f"  {temp} {power}{extra}"
        )
    total_used = sum(npu.hbm_used for npu in npus)
    total_mem = sum(npu.hbm_total for npu in npus)
    avg_aicore = sum(npu.aicore for npu in npus) / len(npus)
    max_temp = max(npu.temp for npu in npus)
    total_power = sum(npu.power for npu in npus)
    hbm_pct = total_used * 100.0 / total_mem if total_mem else 0.0
    lines.append(
        f"  {t('total')}  {t('avg_aicore')} {avg_aicore:3.0f}%"
        f"   {t('hbm')} {hbm_pct:5.1f}% ({total_used / 1024:5.1f}/{total_mem / 1024:5.1f}G)"
        f"   {t('max_temp')} {max_temp}C   {t('total_power')} {total_power:.0f}W"
    )
    bad = [npu for npu in npus if npu.health != "OK"]
    if bad:
        lines.append(
            "  " + theme.red("abnormal: " + ", ".join(f"NPU{npu.id}={npu.health}" for npu in bad))
        )
    return lines


def render_chart(npus: Sequence[NpuSnapshot], t: Translator, theme: Theme, width: int) -> List[str]:
    items = list(npus)[:8]
    if not items:
        return []
    metrics = (
        ("aicore", t("aicore"), 100.0, CYAN),
        ("hbm", t("hbm"), 100.0, MAGENTA),
        ("temp", t("temp"), 100.0, YELLOW),
        ("power", t("power"), 300.0, GREEN),
    )
    height = 8
    bar_width = 2 if width >= 120 else 1
    group_width = len(metrics) * bar_width + len(metrics) - 1
    gap = "  "
    lines = [
        f"{theme.bold(t('chart_title'))}  {theme.dim(t('chart_legend'))}",
        f"{theme.dim(t('chart_scale'))}",
    ]
    for level in range(height, 0, -1):
        if level == height or level % 2 == 0:
            axis = f"{level * 100 // height:3d}% ┤"
        else:
            axis = "     │"
        groups = []
        for npu in items:
            values = {
                "aicore": float(npu.aicore),
                "hbm": float(npu.hbm_pct),
                "temp": float(npu.temp),
                "power": float(npu.power),
            }
            blocks = []
            for key, _label, top, color in metrics:
                filled = values[key] * height / top >= level
                block = "█" * bar_width if filled else " " * bar_width
                blocks.append(theme.paint(block, color))
            groups.append(" ".join(blocks))
        lines.append(axis + " " + gap.join(groups))
    chart_width = len(items) * group_width + (len(items) - 1) * len(gap)
    lines.append("     └" + "─" * (chart_width + 1))
    labels = [("#" + str(npu.id)).center(group_width) for npu in items]
    lines.append("       " + gap.join(labels))
    return lines


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "npu-top" / "config.json"


def load_config(path: Optional[str], explicit: bool) -> Dict[str, object]:
    config_path = Path(path).expanduser() if path else default_config_path()
    if not config_path.exists():
        if explicit:
            raise SystemExit(f"npu-top: config not found: {config_path}")
        return {}
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"npu-top: cannot read {config_path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"npu-top: invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"npu-top: config root must be a JSON object: {config_path}")
    return data


def build_endpoints(args: argparse.Namespace, config: Dict[str, object]) -> List[Endpoint]:
    raw = args.endpoint if args.endpoint else config.get("endpoints")
    if raw is None:
        items: List[object] = [DEFAULT_ENDPOINT]
    elif isinstance(raw, list):
        items = list(raw)
    else:
        raise SystemExit("npu-top: endpoints must be a list")
    if not items:
        raise SystemExit("npu-top: at least one endpoint is required")

    try:
        global_max = (
            args.max_seqs
            if args.max_seqs is not None
            else int(config.get("max_seqs", 0) or 0)
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit("npu-top: max_seqs must be an integer") from exc
    if global_max < 0:
        raise SystemExit("npu-top: max_seqs must be >= 0")

    global_key = args.api_key or os.environ.get("NPU_TOP_API_KEY", "")
    if not global_key and config.get("api_key_env"):
        global_key = os.environ.get(str(config["api_key_env"]), "")

    endpoints: List[Endpoint] = []
    seen: Dict[str, int] = collections.defaultdict(int)
    for item in items:
        if isinstance(item, str):
            try:
                name, url = parse_endpoint_spec(item)
            except ValueError as exc:
                raise SystemExit(f"npu-top: {exc}") from exc
            key = global_key
            max_seqs = global_max
        elif isinstance(item, dict):
            raw_url = item.get("url")
            if not isinstance(raw_url, str) or not raw_url.strip():
                raise SystemExit("npu-top: every config endpoint needs a url")
            try:
                url = normalize_metrics_url(raw_url)
            except ValueError as exc:
                raise SystemExit(f"npu-top: {exc}") from exc
            name = str(item.get("name") or label_from_url(url)).strip()
            key = ""
            env_name = item.get("api_key_env")
            if env_name:
                key = os.environ.get(str(env_name), "")
            if args.api_key:
                key = args.api_key
            elif not key:
                key = global_key
            try:
                max_seqs = (
                    args.max_seqs
                    if args.max_seqs is not None
                    else int(item.get("max_seqs", global_max) or 0)
                )
            except (TypeError, ValueError) as exc:
                raise SystemExit("npu-top: endpoint max_seqs must be an integer") from exc
        else:
            raise SystemExit("npu-top: endpoints must contain strings or objects")
        if max_seqs < 0:
            raise SystemExit(f"npu-top: max_seqs for {name} must be >= 0")
        seen[name] += 1
        if seen[name] > 1:
            name = f"{name}-{seen[name]}"
        endpoints.append(Endpoint(name=name, url=url, api_key=key, max_seqs=max_seqs))
    return endpoints


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npu-top",
        description="Tiny zero-dependency dashboard for vLLM metrics and Ascend NPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  npu-top\n"
            "  npu-top -e deepseek=http://127.0.0.1:8900 --max-seqs 32 --spark --throughput-chart\n"
            "  npu-top -e a=http://127.0.0.1:8000 -e b=http://127.0.0.1:8001\n"
            "  npu-top --once --json\n"
            "  npu-top --no-npu --once\n"
        ),
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        action="append",
        metavar="[NAME=]URL",
        help="vLLM metrics URL; repeat for multiple services (default: %s)" % DEFAULT_ENDPOINT,
    )
    parser.add_argument("--config", metavar="PATH", help="JSON config (default: ~/.config/npu-top/config.json)")
    parser.add_argument("--api-key", help="vLLM bearer token; prefer NPU_TOP_API_KEY")
    parser.add_argument("--max-seqs", type=nonnegative_int, help="vLLM --max-num-seqs, for the concurrency bar")
    parser.add_argument("--interval", type=positive_float, help="refresh interval in seconds (default: 1.0)")
    parser.add_argument("--timeout", type=positive_float, help="HTTP timeout in seconds (default: 3.0)")
    parser.add_argument("--once", action="store_true", help="print one frame and exit")
    parser.add_argument("--json", action="store_true", help="print one JSON snapshot and exit")
    parser.add_argument("--no-warmup", action="store_true", help="skip the second sample used for rates")
    parser.add_argument("--spark", action="store_true", help="show sparklines")
    parser.add_argument(
        "--throughput-chart",
        action="store_true",
        help="show input/generation throughput bar charts",
    )
    parser.add_argument("--chart", action="store_true", help="show the 8-NPU grouped bar chart")
    parser.add_argument("--no-npu", action="store_true", help="skip npu-smi")
    parser.add_argument("--npu-smi", metavar="PATH", help="path to npu-smi (default: search PATH)")
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--color", action="store_true", help="force ANSI colors (useful for screenshots)")
    color_group.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument("--lang", choices=("auto", "en", "zh"), default="auto", help="UI language (default: auto)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


def configure_redirected_stdout() -> None:
    if os.name != "nt" or sys.stdout.isatty():
        return
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8")
    except Exception:
        pass


def terminal_width() -> int:
    return max(64, min(140, shutil.get_terminal_size((100, 30)).columns))


def frame_payload(
    services: Sequence[ServiceSnapshot], npus: Sequence[NpuSnapshot], npu_error: str
) -> Dict[str, object]:
    def service_dict(item: ServiceSnapshot) -> Dict[str, object]:
        return {
            "name": item.name,
            "url": item.url,
            "online": item.online,
            "error": item.error,
            "model": item.model,
            "max_seqs": item.max_seqs,
            "running": item.running,
            "waiting": item.waiting,
            "swapped": item.swapped,
            "kv_cache_usage": round(item.kv, 4),
            "prompt_tokens_per_second": round(item.prompt_tps, 3),
            "generation_tokens_per_second": round(item.gen_tps, 3),
            "requests_per_second": round(item.req_tps, 4),
            "ttft_seconds": round(item.ttft, 6),
            "tpot_seconds": round(item.tpot, 6),
            "e2e_seconds": round(item.e2e, 6),
            "queue_seconds": round(item.queue, 6),
            "prefill_seconds": round(item.prefill, 6),
            "decode_seconds": round(item.decode, 6),
            "itl_seconds": round(item.itl, 6),
            "preemptions": item.preempt,
            "spec_decode_accept_percent": round(item.mtp, 2),
            "success_total": item.success_total,
            "prompt_tokens_total": item.prompt_total,
            "generation_tokens_total": item.gen_total,
        }

    def npu_dict(item: NpuSnapshot) -> Dict[str, object]:
        return {
            "id": item.id,
            "name": item.name,
            "health": item.health,
            "aicore_percent": item.aicore,
            "hbm_used_mb": item.hbm_used,
            "hbm_total_mb": item.hbm_total,
            "hbm_percent": round(item.hbm_pct, 2),
            "temperature_c": item.temp,
            "power_w": item.power,
        }

    return {
        "tool": "npu-top",
        "version": __version__,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "services": [service_dict(item) for item in services],
        "npus": [npu_dict(item) for item in npus],
        "npu_error": npu_error or None,
    }


def render_frame(
    services: Sequence[ServiceSnapshot],
    npus: Sequence[NpuSnapshot],
    npu_error: str,
    monitor: Monitor,
    t: Translator,
    theme: Theme,
    args: argparse.Namespace,
    interval: float,
    interactive: bool,
) -> str:
    width = terminal_width()
    lines = [
        f"{theme.bold('npu-top')} {__version__}  {t('subtitle')}  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "─" * width,
    ]
    for service in services:
        lines.extend(render_service(service, t, theme, args.spark, width))
        if args.throughput_chart:
            chart = render_throughput_chart(service.throughput_history, t, theme, width)
            if chart:
                lines.append("")
                lines.extend(chart)
        lines.append("")
    lines.extend(render_npus(npus, npu_error, monitor.npu_history, t, theme, args.spark))
    if args.chart:
        chart = render_chart(npus, t, theme, width)
        if chart:
            lines.append("")
            lines.extend(chart)
    if interactive:
        lines.extend(["", theme.dim(t("quit_hint", interval=interval))])
    return "\n".join(lines)


def wait_for_key(seconds: float) -> bool:
    if os.name == "nt":
        import msvcrt

        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key.lower() == "q" or key == "\x03":
                    return True
            time.sleep(0.05)
        return False
    ready, _, _ = select.select([sys.stdin], [], [], seconds)
    if not ready:
        return False
    key = sys.stdin.read(1)
    return key.lower() == "q" or key == "\x03" or key == ""


def run_tui(
    monitor: Monitor,
    t: Translator,
    theme: Theme,
    args: argparse.Namespace,
    interval: float,
) -> int:
    restore = None
    if os.name != "nt":
        try:
            import termios
            import tty

            old = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            restore = (termios, old)
        except Exception:
            restore = None

    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    try:
        while True:
            services, npus, npu_error = monitor.poll()
            frame = render_frame(
                services, npus, npu_error, monitor, t, theme, args, interval, True
            )
            sys.stdout.write("\033[H\033[2J" + frame)
            sys.stdout.flush()
            if wait_for_key(interval):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if restore is not None:
            module, old = restore
            module.tcsetattr(sys.stdin.fileno(), module.TCSADRAIN, old)
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_redirected_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config, bool(args.config))
    lang = resolve_lang(args.lang)
    t = Translator(lang)
    theme = Theme(enabled=args.color or (not args.no_color and supports_color()))

    try:
        endpoints = build_endpoints(args, config)
        interval = float(
            args.interval if args.interval is not None else config.get("interval", 1.0)
        )
        timeout = float(
            args.timeout if args.timeout is not None else config.get("timeout", 3.0)
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    interval = max(0.2, interval)
    if timeout <= 0:
        parser.error("--timeout must be > 0")

    npu_enabled = not args.no_npu and config.get("npu", True) is not False
    npu_binary = args.npu_smi or config.get("npu_smi") or shutil.which("npu-smi")
    monitor = Monitor(
        endpoints,
        npu_binary=str(npu_binary) if npu_binary else None,
        timeout=timeout,
        npu_enabled=npu_enabled,
    )

    def one_shot() -> Tuple[List[ServiceSnapshot], List[NpuSnapshot], str]:
        services, npus, npu_error = monitor.poll()
        if not args.no_warmup:
            time.sleep(min(interval, 1.0))
            services, npus, npu_error = monitor.poll()
        return services, npus, npu_error

    if args.json:
        services, npus, npu_error = one_shot()
        print(json.dumps(frame_payload(services, npus, npu_error), ensure_ascii=False, indent=2))
        return 0

    if args.once or not (sys.stdout.isatty() and sys.stdin.isatty()):
        services, npus, npu_error = one_shot()
        print(render_frame(services, npus, npu_error, monitor, t, theme, args, interval, False))
        return 0

    return run_tui(monitor, t, theme, args, interval)


if __name__ == "__main__":
    raise SystemExit(main())
