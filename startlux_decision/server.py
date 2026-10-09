"""TypeSafe-compatible decision server.

    python -m startlux_decision.server --model /path/to/StartLux-Decision-4B --port 8090

POST /v1/systemone  {"state": ..., "questions": {key: {"type", "instructions", "criteria"}}, "images": [...]}
  -> {"answers": {key: answer}, "usage": {"input_tokens", "output_tokens"}, "model": ...}
     "images" is optional: base64 strings or data URIs, part of the evidence ("<image>" in a string state marks where
     each goes, otherwise they come first); the torch backend only.
GET  /health        {"status", "model", "backend", "accelerator", "fast_kernels", "graphs", "cuda_graphs", ...}
GET  /v1/models     {"models": [{"name", "description", "release_date"}]}
Every response carries an x-typesafe-request-id header; errors are {"error": message, "detail": [{"loc", "msg", "type"}]}
(400 for a malformed body, 422 for a request the model cannot answer).  confidence follows TypeSafe's definitions.
Requests are served one at a time on one GPU.  At start-up one request runs through both the GPU-graph path and the
eager path; if their probabilities differ by more than 0.02 the graphs are dropped.

--backend auto (the default) runs the model with MLX on Apple Silicon when mlx-lm is installed (mlx_model.py) and with
PyTorch everywhere else; --int8 adds int8 matmuls on M5 and later Macs (mlx_int8.py).
"""
import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def mlx_available():
    """True on Apple Silicon with mlx-lm installed (requirements.txt installs it there)."""
    try:
        import mlx.core as mx
        import mlx_lm  # noqa: F401
        return mx.metal.is_available()
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local StartLux-Decision directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--device", help="cuda (NVIDIA CUDA or AMD ROCm) or cpu for torch (default: cuda when available)")
    ap.add_argument("--backend", choices=("auto", "torch", "mlx"), default="auto",
                    help="auto: MLX on Apple Silicon when mlx-lm is installed, else torch")
    ap.add_argument("--int8", action="store_true", help="MLX on M5 and later: int8 matmuls on the neural accelerators")
    ap.add_argument("--name", help="model name reported in responses (default: the directory name)")
    ap.add_argument("--max-length", type=int, default=262144, help="longest prompt in tokens (torch backend)")
    ap.add_argument("--max-pixels", type=int, default=1 << 20, help="pixels per image after resizing (torch backend)")
    ap.add_argument("--no-images", action="store_true", help="leave the vision tower out (torch backend)")
    a = ap.parse_args()
    backend = a.backend
    if backend == "auto":
        backend = "mlx" if a.device is None and mlx_available() else "torch"
    if backend == "mlx":
        from .mlx_model import MLXDecision
        engine = MLXDecision(a.model, int8=a.int8)
        engine.warm_up()
    else:
        from .model import StartLuxDecision
        engine = StartLuxDecision(a.model, device=a.device, max_length=a.max_length, images=not a.no_images,
                                  max_pixels=a.max_pixels)
    accelerator = "metal" if backend == "mlx" else engine.accelerator
    demo_state = {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."}
    demo_questions = {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                              "technical": "App, login and account problems"}},
        "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
        "severity": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["cosmetic", "annoying", "blocks the customer"]},
    }
    engine.decide(demo_state, demo_questions)                     # warm-up
    if engine.graphs:
        diff = engine.self_test(demo_state, demo_questions)
        print(f"graph self-test: max |p_graph - p_eager| = {diff:.2e}", flush=True)
        if diff > 0.02:
            print("graph readout disagrees with the eager path, graphs disabled", flush=True)
            engine.graphs = {}
    lock = threading.Lock()
    name = a.name or os.path.basename(os.path.normpath(a.model))
    try:
        released = json.load(open(os.path.join(a.model, "decision_config.json"))).get("release_date", "2026-10-01")
    except (OSError, ValueError):
        released = "2026-10-01"

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("x-typesafe-request-id", uuid.uuid4().hex)
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code, message):
            self._send(code, {"error": message, "detail": [{"loc": ["body"], "msg": message, "type": "value_error"}]})

        def do_GET(self):
            path = self.path.rstrip("/")
            if path in ("/health", "/v1/health"):
                return self._send(200, {"status": "ok", "model": name, "backend": backend, "accelerator": accelerator,
                                        "fast_kernels": engine.fast_kernels, "graphs": len(engine.graphs),
                                        "cuda_graphs": len(engine.graphs),
                                        "images": getattr(engine, "vision", None) is not None,
                                        "max_length": getattr(engine, "max_length", None)})
            if path == "/v1/models":
                return self._send(200, {"models": [{"name": name, "description": "StartLux-Decision typed decision model",
                                                    "release_date": released}]})
            self._error(404, "not found")

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._error(404, "not found")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                questions = body.get("questions")
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty object")
                images = body.get("images") or None
                if images is not None:
                    if not isinstance(images, list) or not all(isinstance(x, str) for x in images):
                        raise ValueError("images must be a list of base64 strings or data URIs")
                    if backend != "torch":
                        raise ValueError("images need the torch backend")
                    from .model import load_image
                    images = [load_image(x, paths=False) for x in images]
            except ValueError as e:
                return self._error(400, str(e))
            t = time.perf_counter()
            try:
                with lock:
                    answers, usage = (engine.decide(body.get("state"), questions, images=images) if images else
                                      engine.decide(body.get("state"), questions))
            except ValueError as e:          # unknown question type, prompt over the context limit
                return self._error(422, str(e))
            self._send(200, {"answers": answers, "usage": usage, "model": name,
                             "latency_ms": round(1000 * (time.perf_counter() - t), 2)})

        def log_message(self, *args):
            pass

    detail = (f"MLX, int8 projections: {engine.int8}" if backend == "mlx" else
              f"accelerator: {accelerator}, fast kernels: {engine.fast_kernels}, graphs: {len(engine.graphs)}, "
              f"images: {'yes' if engine.vision is not None else 'no, ' + engine.vision_note}, "
              f"max length: {engine.max_length}")
    print(f"{name} serving on http://{a.host}:{a.port}/v1/systemone ({detail})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
