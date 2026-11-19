import argparse
import contextlib
import http.server
import io
import json
import os
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import npu_top


METRICS_V1 = """\
# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="deepseek-v4-flash"} 7
vllm:num_requests_waiting{model_name="deepseek-v4-flash"} 0
vllm:kv_cache_usage_perc{model_name="deepseek-v4-flash"} 0.072
vllm:prompt_tokens_total{model_name="deepseek-v4-flash"} 138620000
vllm:generation_tokens_total{model_name="deepseek-v4-flash"} 4530000
vllm:request_success_total{model_name="deepseek-v4-flash"} 6500
vllm:num_preemptions_total{model_name="deepseek-v4-flash"} 2
vllm:spec_decode_num_draft_tokens_total{model_name="deepseek-v4-flash"} 1000
vllm:spec_decode_num_accepted_tokens_total{model_name="deepseek-v4-flash"} 957
vllm:time_to_first_token_seconds_count{model_name="deepseek-v4-flash"} 100
vllm:time_to_first_token_seconds_sum{model_name="deepseek-v4-flash"} 204
vllm:e2e_request_latency_seconds_count{model_name="deepseek-v4-flash"} 100
vllm:e2e_request_latency_seconds_sum{model_name="deepseek-v4-flash"} 6844
vllm:request_queue_time_seconds_count{model_name="deepseek-v4-flash"} 100
vllm:request_queue_time_seconds_sum{model_name="deepseek-v4-flash"} 0
vllm:request_time_per_output_token_seconds_count{model_name="deepseek-v4-flash"} 100
vllm:request_time_per_output_token_seconds_sum{model_name="deepseek-v4-flash"} 3.9
"""

METRICS_V2 = METRICS_V1.replace(
    "vllm:prompt_tokens_total{model_name=\"deepseek-v4-flash\"} 138620000",
    "vllm:prompt_tokens_total{model_name=\"deepseek-v4-flash\"} 138621000",
).replace(
    "vllm:generation_tokens_total{model_name=\"deepseek-v4-flash\"} 4530000",
    "vllm:generation_tokens_total{model_name=\"deepseek-v4-flash\"} 4530200",
).replace(
    "vllm:request_success_total{model_name=\"deepseek-v4-flash\"} 6500",
    "vllm:request_success_total{model_name=\"deepseek-v4-flash\"} 6510",
).replace(
    "vllm:num_preemptions_total{model_name=\"deepseek-v4-flash\"} 2",
    "vllm:num_preemptions_total{model_name=\"deepseek-v4-flash\"} 3",
).replace(
    "vllm:time_to_first_token_seconds_count{model_name=\"deepseek-v4-flash\"} 100",
    "vllm:time_to_first_token_seconds_count{model_name=\"deepseek-v4-flash\"} 110",
).replace(
    "vllm:time_to_first_token_seconds_sum{model_name=\"deepseek-v4-flash\"} 204",
    "vllm:time_to_first_token_seconds_sum{model_name=\"deepseek-v4-flash\"} 214",
).replace(
    "vllm:spec_decode_num_draft_tokens_total{model_name=\"deepseek-v4-flash\"} 1000",
    "vllm:spec_decode_num_draft_tokens_total{model_name=\"deepseek-v4-flash\"} 1100",
).replace(
    "vllm:spec_decode_num_accepted_tokens_total{model_name=\"deepseek-v4-flash\"} 957",
    "vllm:spec_decode_num_accepted_tokens_total{model_name=\"deepseek-v4-flash\"} 1052",
)


NPU_INFO = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc3                 Version: 23.0.rc3                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2               | OK            | 92.3        36                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          479    / 65536       |
+===========================+===============+====================================================+
| 1     910B2               | Warning       | N/A         67                0    / 0             |
| 1                         | 0000:C2:00.0  | 84          0    / 0          61000  / 65536       |
+===========================+===============+====================================================+
"""


class ParseTests(unittest.TestCase):
    def test_parse_prometheus_and_sum(self):
        samples = npu_top.parse_prometheus(METRICS_V1)
        self.assertEqual(len(samples["vllm:num_requests_running"]), 1)
        self.assertEqual(
            npu_top.metric_sum(samples, "vllm:num_requests_running"), 7.0
        )
        self.assertEqual(npu_top.metric_model(samples), "deepseek-v4-flash")

    def test_endpoint_normalization(self):
        self.assertEqual(
            npu_top.normalize_metrics_url("127.0.0.1:8000"),
            "http://127.0.0.1:8000/metrics",
        )
        self.assertEqual(
            npu_top.normalize_metrics_url("https://host/metrics"),
            "https://host/metrics",
        )
        name, url = npu_top.parse_endpoint_spec("deepseek=http://127.0.0.1:8900")
        self.assertEqual(name, "deepseek")
        self.assertEqual(url, "http://127.0.0.1:8900/metrics")

    def test_snapshot_rates(self):
        endpoint = npu_top.Endpoint("ds", "http://127.0.0.1:8900/metrics", max_seqs=32)
        first, flat = npu_top.build_snapshot(endpoint, METRICS_V1, None, 1.0, {})
        self.assertEqual(first.model, "deepseek-v4-flash")
        self.assertEqual(first.running, 7)
        self.assertAlmostEqual(first.kv, 0.072)
        self.assertAlmostEqual(first.ttft, 2.04)
        self.assertAlmostEqual(first.tpot, 0.039)
        self.assertAlmostEqual(first.mtp, 95.7)

        second, _ = npu_top.build_snapshot(endpoint, METRICS_V2, flat, 10.0, {})
        self.assertAlmostEqual(second.prompt_tps, 100.0)
        self.assertAlmostEqual(second.gen_tps, 20.0)
        self.assertAlmostEqual(second.req_tps, 1.0)
        self.assertEqual(second.preempt, 1)
        self.assertAlmostEqual(second.ttft, 1.0)

    def test_npu_parser(self):
        npus = npu_top.parse_npu_info(NPU_INFO)
        self.assertEqual([npu.id for npu in npus], [0, 1])
        self.assertEqual(npus[0].name, "910B2")
        self.assertEqual(npus[0].health, "OK")
        self.assertEqual(npus[0].aicore, 0)
        self.assertEqual(npus[0].hbm_used, 479)
        self.assertEqual(npus[0].hbm_total, 65536)
        self.assertAlmostEqual(npus[0].hbm_pct, 479 * 100 / 65536)
        self.assertEqual(npus[1].power, 0.0)
        self.assertEqual(npus[1].temp, 67)
        self.assertEqual(npus[1].aicore, 84)

    def test_sparkline(self):
        line = npu_top.sparkline([0, 50, 100], 3, ceiling=100)
        self.assertEqual(len(line), 3)
        self.assertTrue(line.startswith(npu_top.SPARK[0]))
        self.assertTrue(line.endswith(npu_top.SPARK[-1]))


class ConfigTests(unittest.TestCase):
    def test_build_endpoints_from_cli(self):
        args = argparse.Namespace(
            endpoint=["a=http://127.0.0.1:8000", "b=http://127.0.0.1:8001"],
            max_seqs=32,
            api_key="",
        )
        endpoints = npu_top.build_endpoints(args, {})
        self.assertEqual([item.name for item in endpoints], ["a", "b"])
        self.assertEqual([item.max_seqs for item in endpoints], [32, 32])

    def test_build_endpoints_from_config_and_dedupe(self):
        old = os.environ.get("TEST_VLLM_KEY")
        os.environ["TEST_VLLM_KEY"] = "secret"
        try:
            args = argparse.Namespace(endpoint=None, max_seqs=None, api_key="")
            endpoints = npu_top.build_endpoints(
                args,
                {
                    "endpoints": [
                        {"name": "a", "url": "http://127.0.0.1:8000", "api_key_env": "TEST_VLLM_KEY"},
                        {"name": "a", "url": "http://127.0.0.1:8001"},
                    ]
                },
            )
        finally:
            if old is None:
                os.environ.pop("TEST_VLLM_KEY", None)
            else:
                os.environ["TEST_VLLM_KEY"] = old
        self.assertEqual([item.name for item in endpoints], ["a", "a-2"])
        self.assertEqual(endpoints[0].api_key, "secret")


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        body = METRICS_V1.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def start_metrics_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


class FetchTests(unittest.TestCase):
    def test_fetch_metrics_http(self):
        server, thread = start_metrics_server()
        try:
            endpoint = npu_top.Endpoint(
                "mock",
                f"http://127.0.0.1:{server.server_address[1]}/metrics",
            )
            text = npu_top.fetch_metrics(endpoint, 2.0)
            self.assertIn("vllm:num_requests_running", text)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class MainTests(unittest.TestCase):
    def test_main_json_no_npu(self):
        server, thread = start_metrics_server()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = npu_top.main(
                    [
                        "--json",
                        "--no-npu",
                        "--interval",
                        "0.2",
                        "-e",
                        f"mock={url}",
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["services"][0]["online"])
            self.assertIn("generation_tokens_per_second", payload["services"][0])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class RenderTests(unittest.TestCase):
    def test_frame_payload_is_json_serializable_and_has_no_history(self):
        endpoint = npu_top.Endpoint("ds", "http://127.0.0.1:8900/metrics", max_seqs=32)
        snapshot, _ = npu_top.build_snapshot(endpoint, METRICS_V1, None, 1.0, {})
        snapshot.history = [(1, 2.0)]
        npus = npu_top.parse_npu_info(NPU_INFO)
        payload = npu_top.frame_payload([snapshot], npus, "")
        text = json.dumps(payload)
        self.assertIn('"tool": "npu-top"', text)
        self.assertNotIn("history", text)
        self.assertEqual(len(payload["npus"]), 2)

    def test_render_service_offline(self):
        snapshot = npu_top.ServiceSnapshot(
            name="ds", url="http://127.0.0.1:8900/metrics", online=False, error="refused"
        )
        lines = npu_top.render_service(
            snapshot,
            npu_top.Translator("en"),
            npu_top.Theme(enabled=False),
        )
        self.assertIn("offline: refused", "\n".join(lines))

    def test_render_throughput_chart(self):
        history = [(100.0 + index * 25, 20.0 + index * 3) for index in range(20)]
        lines = npu_top.render_throughput_chart(
            history,
            npu_top.Translator("en"),
            npu_top.Theme(enabled=False),
            100,
        )
        text = "\n".join(lines)
        self.assertIn("Throughput bar chart", text)
        self.assertIn("prompt", text)
        self.assertIn("generation", text)
        self.assertIn("peak", text)
        self.assertIn("█", text)
        self.assertIn("oldest", text)
        self.assertIn("now", text)


if __name__ == "__main__":
    unittest.main()
