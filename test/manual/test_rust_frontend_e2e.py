"""Opt-in, real-model validation; see rust_frontend_e2e.md for the three runs."""

import importlib
import importlib.metadata
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import requests
from _rust_frontend_e2e_client import (
    PROMPT,
    grpc_frames,
    http_frames,
    make_body,
    summarize,
)

import sglang
from sglang.srt.rust_extensions import load_rust_extension
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

APIS = ("text", "tokens", "completion", "chat")
PATHS = dict(
    text="/generate",
    tokens="/generate",
    completion="/v1/completions",
    chat="/v1/chat/completions",
)


def wait_for(predicate, description, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"Timed out: {description}")


def stable(value):
    """Strip only generated IDs/timing, not output, usage, errors or finish reasons."""
    if isinstance(value, dict):
        return {
            k: stable(v)
            for k, v in value.items()
            if k not in ("id", "created", "e2e_latency")
        }
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


class TestRustFrontendE2E(CustomTestCase):
    process = None
    owned_children = []
    channel = None
    server_log = None
    tests_succeeded = False

    @classmethod
    def setUpClass(cls):
        if not __debug__:
            raise RuntimeError("Do not run this suite with Python -O")
        os.environ["SGLANG_TEST_MAX_RETRY"] = "0"
        cls.output = Path(os.environ["SGLANG_E2E_OUTPUT_DIR"]).resolve()
        cls.output.mkdir(parents=True, exist_ok=False)
        cls.model = str(Path(os.environ["SGLANG_E2E_MODEL_PATH"]).resolve(strict=True))
        cls.dual = os.environ.get("SGLANG_E2E_HTTP_ONLY") != "1"
        cls.base_url = (
            f"http://127.0.0.1:{int(os.environ.get('SGLANG_E2E_HTTP_PORT', '30000'))}"
        )
        cls.grpc_port = int(os.environ.get("SGLANG_E2E_GRPC_PORT", "50051"))
        cls.source = Path(sglang.__file__).resolve().parents[2]
        assert (cls.source / "rust/Cargo.toml").is_file(), (
            "Import SGLang from the source checkout"
        )
        launcher = Path(shutil.which("sglang") or "missing-sglang-command")
        interpreter = launcher.read_text().splitlines()[0].removeprefix("#!")
        assert Path(interpreter).absolute() == Path(sys.executable).absolute(), (
            "Run this test with the interpreter named by the sglang executable"
        )
        extension = load_rust_extension(
            "sglang.srt.rust_extensions._server", mode="auto"
        )
        cls.tokenizer = get_tokenizer(cls.model)
        cls.tokens = cls.tokenizer.encode(PROMPT)
        revision = (
            os.environ.get("SGLANG_E2E_SOURCE_REVISION")
            or subprocess.check_output(
                ["git", "-C", str(cls.source), "rev-parse", "HEAD"], text=True
            ).strip()
        )
        cls.report = {
            "source_revision": revision,
            "source": str(cls.source),
            "extension": extension.__file__,
            "model": cls.model,
            "dual_protocol": cls.dual,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "sglang-kernel", "grpcio")
            },
            "gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            "http": {},
            "completed_tests": [],
            "passed": False,
        }
        if cls.dual:
            import grpc

            cls.grpc = grpc
            generated = cls.output / "client"
            generated.mkdir()
            proto = cls.source / "proto/sglang/runtime/v1"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "grpc_tools.protoc",
                    f"-I{proto}",
                    f"--python_out={generated}",
                    f"--grpc_python_out={generated}",
                    str(proto / "sglang.proto"),
                ],
                check=True,
            )
            sys.path.insert(0, str(generated))
            cls.pb = importlib.import_module("sglang_pb2")
            cls.stub_type = importlib.import_module("sglang_pb2_grpc").SglangServiceStub
        cls.launch()

    @classmethod
    def launch(cls):
        offset = len(cls.logs()) if (cls.output / "server.log").exists() else 0
        cls.server_log = (cls.output / "server.log").open("a")
        args = [
            "--served-model-name",
            "e2e-model",
            "--random-seed",
            "42",
            "--context-length",
            "4096",
            "--mem-fraction-static",
            "0.5",
            "--max-running-requests",
            "2",
            "--disable-cuda-graph",
            "--disable-prefill-cuda-graph",
            "--disable-radix-cache",
            "--log-level",
            "debug",
        ]
        if cls.dual:
            args += ["--grpc-port", str(cls.grpc_port)]
        cls.report["server_args"] = args
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            return_stdout_stderr=(cls.server_log, cls.server_log),
            env={
                "SGLANG_RUST_SERVER": "1",
                "SGLANG_RUST_BUILD_MODE": "auto",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
                "PYTHONPATH": str(cls.source / "python"),
            },
        )
        cls.owned_children = psutil.Process(cls.process.pid).children(recursive=True)
        wait_for(
            lambda: "The server is fired up and ready to roll!" in cls.logs()[offset:],
            "model warmup",
            timeout=120,
        )
        assert "SGLANG_RUST_SERVER enabled" in cls.logs()[offset:], (
            "Not the Rust frontend"
        )
        if cls.dual:
            assert "gRPC server listening" in cls.logs()[offset:], (
                "Not the Rust Tonic listener"
            )
            cls.channel = cls.grpc.insecure_channel(f"127.0.0.1:{cls.grpc_port}")
            cls.grpc.channel_ready_future(cls.channel).result(timeout=15)
            cls.stub = cls.stub_type(cls.channel)

    @classmethod
    def logs(cls):
        return (cls.output / "server.log").read_text(errors="replace")

    @classmethod
    def tearDownClass(cls):
        cleaned_up = False
        try:
            if cls.channel is not None:
                cls.channel.close()
            if cls.process is not None and cls.process.poll() is None:
                kill_process_tree(cls.process.pid)
            # The launcher may have exited without cleaning up its children.
            for child in cls.owned_children:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
            if cls.server_log is not None:
                cls.server_log.close()
            cleaned_up = True
        finally:
            if hasattr(cls, "report"):
                cls.report["passed"] = cleaned_up and cls.tests_succeeded
                (cls.output / "report.json").write_text(
                    json.dumps(cls.report, indent=2) + "\n"
                )

    def tearDown(self):
        self.report["completed_tests"].append(self._testMethodName)
        type(self).tests_succeeded = (
            self._outcome.result.wasSuccessful()
            and len(self.report["completed_tests"]) == 6
        )

    def body(self, api, stream):
        return make_body(api, stream, "e2e-model", self.tokens)

    def open_call(self, api, body, transport):
        if transport == "http":
            return requests.post(
                self.base_url + PATHS[api],
                json=body,
                stream=body["stream"],
                timeout=(5, 30),
            )
        if api in ("text", "tokens"):
            body = dict(
                body, sampling_params=self.pb.SamplingParams(**body["sampling_params"])
            )
            request_type, method = (
                (self.pb.TextGenerateRequest, self.stub.TextGenerate)
                if api == "text"
                else (self.pb.GenerateRequest, self.stub.Generate)
            )
            return method(request_type(**body), timeout=30)
        method = self.stub.ChatComplete if api == "chat" else self.stub.Complete
        return method(
            self.pb.OpenAIRequest(json_body=json.dumps(body).encode()), timeout=30
        )

    def frames(self, call, api, stream, transport):
        return (
            http_frames(call, stream)
            if transport == "http"
            else grpc_frames(call, api, stream)
        )

    def generate(self, api, stream, transport="http", body=None):
        call = self.open_call(api, body or self.body(api, stream), transport)
        try:
            frames = list(self.frames(call, api, stream, transport))
            return summarize(api, frames, stream), frames
        finally:
            call.close() if transport == "http" else call.cancel()

    def test_01_generation_matrix_and_http_baseline(self):
        for api in APIS:
            expected = None
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    summary, frames = self.generate(api, stream)
                    self.report["http"][f"{api}/{stream}"] = (
                        summary if stream else stable(frames[0])
                    )
                    if expected is None:
                        expected = summary
                    self.assertEqual(
                        summary, expected, "HTTP stream/nonstream mismatch"
                    )
                    if self.dual:
                        self.assertEqual(self.generate(api, stream, "grpc")[0], summary)
        # Keep a malformed request in the before/after HTTP comparison too.
        with requests.post(
            self.base_url + "/v1/completions",
            json={"model": "e2e-model", "prompt": PROMPT, "n": 0},
            timeout=30,
        ) as response:
            self.assertEqual(response.status_code, 400)
            self.report["http"]["invalid"] = response.json()
        reference = os.environ.get("SGLANG_E2E_REFERENCE")
        if reference:
            baseline = json.loads(Path(reference).read_text())
            self.assertTrue(baseline["passed"], "Reference run did not pass")
            self.assertEqual(baseline["model"], self.model)
            self.assertEqual(baseline["packages"], self.report["packages"])
            self.assertEqual(baseline["gpu"], self.report["gpu"])
            self.assertEqual(
                baseline["server_args"][:-2]
                if baseline["dual_protocol"]
                else baseline["server_args"],
                self.report["server_args"][:-2]
                if self.dual
                else self.report["server_args"],
            )
            self.assertEqual(baseline["http"], self.report["http"])
            self.report["reference"] = reference

    def test_02_metadata_and_health(self):
        for path in ("/health", "/health_generate"):
            self.assertEqual(
                requests.get(self.base_url + path, timeout=30).status_code, 200
            )
        info = requests.get(self.base_url + "/get_model_info", timeout=30).json()
        self.assertEqual(info["served_model_name"], "e2e-model")
        with self.subTest(operation="http/server_info"):
            with requests.get(self.base_url + "/server_info", timeout=30) as response:
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["max_context_length"], 4096)
        if self.dual:
            self.assertTrue(
                self.stub.HealthCheck(self.pb.HealthCheckRequest(), timeout=30).healthy
            )
            self.assertEqual(
                json.loads(
                    self.stub.GetModelInfo(
                        self.pb.GetModelInfoRequest(), timeout=30
                    ).json_info
                ),
                info,
            )
            models = self.stub.ListModels(
                self.pb.ListModelsRequest(), timeout=30
            ).models
            self.assertEqual(
                [(m.id, m.max_model_len) for m in models], [("e2e-model", 4096)]
            )
            with self.subTest(operation="grpc/GetServerInfo"):
                server = json.loads(
                    self.stub.GetServerInfo(
                        self.pb.GetServerInfoRequest(), timeout=30
                    ).json_info
                )
                self.assertEqual(server["max_context_length"], 4096)
            decoded = self.stub.Detokenize(
                self.pb.DetokenizeRequest(tokens=self.tokens), timeout=30
            ).text
            self.assertEqual(
                decoded, self.tokenizer.decode(self.tokens, skip_special_tokens=True)
            )

    def test_03_rejections(self):
        if self.dual:
            with self.assertRaises(self.grpc.RpcError) as caught:
                self.stub.Tokenize(self.pb.TokenizeRequest(text=PROMPT), timeout=30)
            self.assertEqual(
                caught.exception.code(), self.grpc.StatusCode.UNIMPLEMENTED
            )
            for option, code in [
                ({"suffix": "!"}, self.grpc.StatusCode.UNIMPLEMENTED),
                ({"n": 0}, self.grpc.StatusCode.INVALID_ARGUMENT),
            ]:
                with self.assertRaises(self.grpc.RpcError) as caught:
                    list(
                        self.open_call(
                            "completion",
                            self.body("completion", False) | option,
                            "grpc",
                        )
                    )
                self.assertEqual(caught.exception.code(), code)
        self.generate("text", False)

    def test_04_concurrent_clients(self):
        # Different prompts make cross-delivery observable, not merely two successes.
        first = self.body("text", False)
        second = self.body("completion", False) | {"prompt": "The capital of France is"}
        # Distinct forced tokens make cross-delivery observable without depending
        # on identical floating-point results for single vs. batched decoding.
        for params, word in ((first["sampling_params"], " apple"), (second, " banana")):
            token = self.tokenizer.encode(word, add_special_tokens=False)[-1]
            self.assertNotIn(token, self.tokenizer.all_special_ids)
            params["logit_bias"] = {str(token): 100}
        transport = "grpc" if self.dual else "http"
        expected = [
            self.generate("text", False, body=first)[0],
            self.generate("completion", False, transport, second)[0],
        ]
        self.assertNotEqual(expected[0]["output"], expected[1]["output"][0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(self.generate, "text", False, "http", first),
                pool.submit(self.generate, "completion", False, transport, second),
            ]
            self.assertEqual([f.result(timeout=40)[0] for f in pending], expected)

    def assert_cancelled(self, api, body, transport, choices=1):
        offset = len(self.logs())
        call = self.open_call(api, body, transport)
        try:
            seen = set()
            for frame in self.frames(call, api, True, transport):
                if api == "text":
                    self.assertIsNone(
                        frame["meta_info"]["finish_reason"],
                        "Generation finished before cancellation",
                    )
                    if frame["text"]:
                        break
                else:
                    for choice in frame["choices"]:
                        self.assertIsNone(
                            choice.get("finish_reason"),
                            "Choice finished before cancellation",
                        )
                        if choice["text"]:
                            seen.add(choice["index"])
                    if len(seen) == choices:
                        break
            else:
                self.fail("No live generation to cancel")
        finally:
            call.close() if transport == "http" else call.cancel()

        def aborted():
            logs = self.logs()[offset:]
            self.assertNotIn("abort dropped:", logs)
            self.assertNotIn("abort encode failed", logs)
            ids = set(
                re.findall(
                    r"Abort (?:running|queued) request\. req\.rid=['\"]([^'\"]+)", logs
                )
            )
            if "rid" in body:
                ids = {rid for rid in ids if rid.split("#")[0] == body["rid"]}
            return len(ids) == choices

        wait_for(
            aborted, f"{transport}/{api}: scheduler cancellation of {choices} choice(s)"
        )
        self.generate("text", False)

    def test_05_cancellation(self):
        for transport in ("http", "grpc") if self.dual else ("http",):
            body = self.body("text", True)
            body["rid"] = f"cancel-{uuid.uuid4().hex}"
            body["sampling_params"].update(max_new_tokens=2048, ignore_eos=True)
            self.assert_cancelled("text", body, transport)
            # Force a normal token so neither OpenAI choice finishes before cancellation.
            token = self.tokenizer.encode(" apple", add_special_tokens=False)[-1]
            self.assertNotIn(token, self.tokenizer.all_special_ids)
            body = self.body("completion", True) | {
                "n": 2,
                "logit_bias": {str(token): 100},
            }
            self.assertEqual(
                self.generate("completion", True, transport, body)[0]["finish_reason"],
                ["length", "length"],
            )
            body["max_tokens"] = 2048
            self.assert_cancelled("completion", body, transport, choices=2)

    def stop_and_assert(self):
        started = time.monotonic()
        self.process.terminate()
        self.process.wait(timeout=30)
        wait_for(
            lambda: all(
                not p.is_running() or p.status() == psutil.STATUS_ZOMBIE
                for p in self.owned_children
            ),
            "owned scheduler processes exiting",
            timeout=15,
        )
        for port in (
            [int(self.base_url.rsplit(":", 1)[1]), self.grpc_port]
            if self.dual
            else [int(self.base_url.rsplit(":", 1)[1])]
        ):
            with socket.socket() as probe:
                probe.settimeout(1)
                self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0)
        self.report.setdefault("shutdown_seconds", []).append(
            time.monotonic() - started
        )

    def test_06_shutdown_and_restart(self):
        calls = []
        try:
            for transport in ("http", "grpc") if self.dual else ("http",):
                body = self.body("text", True)
                body["sampling_params"].update(max_new_tokens=2048, ignore_eos=True)
                call = self.open_call("text", body, transport)
                reader = self.frames(call, "text", True, transport)
                calls.append((transport, call, reader))
                frame = next(reader)
                self.assertIsNone(frame["meta_info"]["finish_reason"])
            self.stop_and_assert()
            for transport, call, _ in calls:
                try:
                    list(call.iter_content() if transport == "http" else call)
                except (
                    requests.RequestException,
                    self.grpc.RpcError if self.dual else requests.RequestException,
                ):
                    pass  # Bounded cancellation, not a promise to finish generation.
        finally:
            for transport, call, _ in calls:
                call.close() if transport == "http" else call.cancel()
        if self.channel is not None:
            self.channel.close()
        self.server_log.close()
        type(self).launch()
        self.generate("text", False)
        if self.dual:
            self.generate("text", False, "grpc")
        self.stop_and_assert()  # Idle shutdown, after rebinding the same ports.


if __name__ == "__main__":
    unittest.main()
