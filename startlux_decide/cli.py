"""startlux-decide: serve, launch-vllm, bench, eval and quantize (``startlux-decide COMMAND --help``)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import sys
import time

from .controls import FIRST_READS, THOUGHT_GENERATIONS
from .modelcfg import THINK_MODES, load_model_folder, load_tokenizer, think_mode


def _controls_json(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--controls must be a JSON object")
    return value


def _urls(values: list[str] | None) -> list[str]:
    return [u for v in values or [] for u in v.split(",") if u.strip()]


def _gpus(text: str) -> list[str]:
    return [g.strip() for g in text.split(",") if g.strip()]


def _add_vllm_launch(p: argparse.ArgumentParser, prefix: str = "") -> None:
    p.add_argument("--gpus", type=_gpus, default=["0"], help="GPUs, one server each (default 0)")
    p.add_argument(f"--{prefix}base-port", type=int, default=8001, help="port of the first server (default 8001)")
    p.add_argument(f"--{prefix}log-dir", default="vllm_logs", help="server logs (default vllm_logs)")
    p.add_argument(f"--{prefix}python", default=sys.executable, help="the Python that runs vLLM (default: this one)")
    p.add_argument("--quantization", choices=("fp8",), help="vLLM weight quantization (re-measure before use)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.88, help="share of each GPU vLLM may use (0.88)")
    p.add_argument("--no-mtp", action="store_true", help="no MTP speculative decoding (default on vLLM 0.31 or later: "
                                                         "3 draft tokens when the folder holds MTP weights)")
    p.add_argument("--no-images", action="store_true", help="text only (default on vLLM 0.31 or later: images in "
                                                            "the evidence when the folder has a vision tower)")
    p.add_argument("--max-images", type=int, default=8, help="most images per request (default 8)")
    p.add_argument("--max-pixels", type=int, default=1 << 20, help="each image is resized to at most this many "
                                                                   "pixels (default 1,048,576)")
    p.add_argument("--startup-timeout", type=float, default=1800.0, help="seconds to wait for the servers")


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="startlux-decide", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the /v1/systemone server")
    p.add_argument("--model", required=True, help="local model folder")
    p.add_argument("--backend", choices=("vllm", "transformers", "llamacpp", "mlx"), default="vllm")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--name", help="model name in responses (default: the folder name)")
    g = p.add_argument_group("vllm backend")
    g.add_argument("--vllm-url", action="append", help="running vLLM server(s): URL, host:port or port; repeat or "
                                                      "separate with commas")
    g.add_argument("--launch-vllm", action="store_true", help="start the vLLM servers too (see launch-vllm)")
    _add_vllm_launch(g, "vllm-")
    g.add_argument("--waves", dest="waves", action="store_true", default=None,
                   help="send chunked thought requests in waves (default: only to vLLM older than 0.31, where a "
                        "thought joining a running batch can be corrupted)")
    g.add_argument("--no-waves", dest="waves", action="store_false", help="never send thought requests in waves")
    g.add_argument("--wave-max", type=int, default=128, help="most thought requests per wave (default 128)")
    g.add_argument("--wave-gather", type=float, default=0.3, help="seconds a wave waits for more requests (0.3)")
    g = p.add_argument_group("other backends")
    g.add_argument("--llamacpp-url", default="http://127.0.0.1:8081", help="llama-server URL")
    g.add_argument("--device", help="transformers: cuda, cpu, ... (default: cuda when available)")
    g.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16", help="transformers weights")
    g = p.add_argument_group("defaults of the decision path (requests may override them)")
    g.add_argument("--think-mode", type=think_mode, metavar="{" + ",".join(THINK_MODES) + "}",
                   help="default: the folder's (auto for AutoThink, never for Instant)")
    g.add_argument("--gate", type=float, help="think when the first top probability is below this (default 0.6; the published numbers used 0.9)")
    g.add_argument("--mix", type=float, help="weight of the read-out after the thought (default 0.75)")
    g.add_argument("--think-budget", type=int, help="thought tokens (default 4096)")
    g.add_argument("--thought-generation", choices=THOUGHT_GENERATIONS,
                   help="auto (default): one streamed request per thought on engines that stream thoughts safely "
                        "(vLLM 0.31 or later), 1,024-token chunks elsewhere; stream; chunked")
    g.add_argument("--first-read", choices=FIRST_READS,
                   help="plain (default): the trained one-pass prompt; thinking: in requests that may think, the "
                        "thinking prompt with an empty thought, so that a thought reuses its prefix cache instead of "
                        "reading the state again (docs/api.md)")
    g.add_argument("--max-think-budget", type=int, help="largest think_budget a request may ask for (default: the "
                                                        "default budget)")
    g.add_argument("--no-none-check", action="store_true",
                   help="do not add the internal \"None of the above\" option to choice questions by default; answers "
                        "then carry no none_probability unless a request sets none_check or allow_none")
    g.add_argument("--allow-none", action="store_true",
                   help="let \"choice\" be null by default when \"None of the above\" is the most likely option; a "
                        "request or a question can still send allow_none false")
    g.add_argument("--max-model-len", type=int, help="context: with --launch-vllm the servers' (default on vLLM 0.31: "
                                                     "262,144 tokens, Instant 131,072); else what the engine reports")
    g = p.add_argument_group("serving publicly (docs/serving_publicly.md)")
    g.add_argument("--api-keys-file", help="require Authorization: Bearer <key> on every route but /health; one key "
                                           "per line, # comments; re-read when the file changes")
    g.add_argument("--ssl-certfile", help="TLS certificate chain (PEM); with --ssl-keyfile serves HTTPS, TLS 1.2+")
    g.add_argument("--ssl-keyfile", help="TLS private key (PEM)")
    g.add_argument("--max-inflight", type=_positive, help="answer 429 at once (Retry-After: 1) when this many "
                                                          "decision requests are in flight (default: no limit)")
    g.add_argument("--openrouter-models-file", help="OpenRouter provider models document (JSON), served as it is; "
                                                    "re-read when the file changes")
    g.add_argument("--openrouter-models-path", default="/openrouter/models", help="route of that document "
                                                                                  "(default /openrouter/models)")
    g.add_argument("--access-log", help="append one JSON line per request: time, request id, route, status, client "
                                        "IP, key id, timings, token counts, number of questions; never content")
    g.add_argument("--request-log", metavar="DIR", help="keep what clients send and get back: one JSON line per "
                                                        "decision request in DIR/requests-YYYY-MM-DD.jsonl with the "
                                                        "client address, the key id, the body and the answers with "
                                                        "their thoughts; images under DIR/images; owner-only files")

    p = sub.add_parser("launch-vllm", help="start vLLM servers with the release flags and keep them running")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--served-model-name", default="m")
    p.add_argument("--max-model-len", type=int, help="context of the servers (default on vLLM 0.31: the folder's "
                                                     "native 262,144 tokens, Instant folders 131,072; else 65,536)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and the environment, start nothing")
    _add_vllm_launch(p)

    p = sub.add_parser("bench", help="latency per request and per phase against a running server")
    p.add_argument("--url", default="http://127.0.0.1:8090")
    p.add_argument("--requests", required=True, help="JSONL, one /v1/systemone body per line")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0, help="untimed requests first")
    p.add_argument("--controls", type=_controls_json,
                   help='JSON merged into every request, e.g. {"think_mode": "never"}')
    p.add_argument("--out", default="bench_summary.csv", help="summary CSV (n, mean, median, p80, p95 per metric)")
    p.add_argument("--rows", help="per-request CSV")
    p.add_argument("--timeout", type=float, default=3600.0)

    p = sub.add_parser("eval", help="accuracy, ECE, Brier and thinking share; or compare two runs")
    p.add_argument("--url", default="http://127.0.0.1:8090")
    p.add_argument("--data", help="JSONL of requests with gold labels")
    p.add_argument("--out", help="predictions JSONL to write")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--controls", type=_controls_json, help="JSON merged into every request")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--score", metavar="PREDICTIONS", help="score a predictions file again")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="paired comparison of two predictions files")
    p.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples of --compare")

    p = sub.add_parser("quantize", help="print or run the documented FP8 / GGUF / MLX steps")
    p.add_argument("method", choices=("fp8", "gguf", "mlx"))
    p.add_argument("--model", required=True)
    p.add_argument("--out", help="output folder (gguf, mlx)")
    p.add_argument("--llama-cpp", help="llama.cpp checkout (gguf)")
    p.add_argument("--type", default="Q8_0", help="GGUF type: BF16, Q8_0, Q4_K_M, ... (default Q8_0)")
    p.add_argument("--bits", type=int, choices=(4, 8), help="MLX quantization bits (default: no quantization)")
    p.add_argument("--run", action="store_true", help="run the commands (default: print them)")
    return ap


# ---------------------------------------------------------------------------------------------- commands

def raise_open_file_limit(want: int = 65536) -> tuple[int, int] | None:
    """Raise the soft limit on open files towards `want`, never above the hard limit. Every request in flight holds a
    client socket and an engine connection, and at the common default of 1,024 a busy server runs out: requests then
    fail with 502 (Too many open files) and asyncio logs a failed accept() in a tight loop. Returns (before, after),
    or None where the platform has no such limit."""
    try:
        import resource
    except ImportError:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = want if hard == resource.RLIM_INFINITY else min(want, hard)
    if soft == resource.RLIM_INFINITY or soft >= target:
        return soft, soft
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        return soft, soft
    return soft, target


def cmd_serve(a: argparse.Namespace) -> int:
    from aiohttp import web

    from .pipeline import default_controls
    from .server import create_app

    limit = raise_open_file_limit()
    if limit and limit[1] != limit[0]:
        logging.info("open-file limit raised from %d to %d", *limit)
    if limit and limit[1] < 8192:
        logging.warning("open-file limit is %d: a busy server may fail with Too many open files; raise the hard "
                        "limit (ulimit -Hn, or LimitNOFILE in systemd)", limit[1])
    options, tls = _public_options(a)
    folder = load_model_folder(a.model)
    if folder.kind == "instant" and a.think_mode not in (None, "never"):
        raise SystemExit("this is an Instant folder: it does not think (drop --think-mode)")
    defaults = default_controls(folder, think_mode=a.think_mode, gate=a.gate, budget=a.think_budget, mix=a.mix,
                                thought_generation=a.thought_generation, first_read=a.first_read,
                                none_check=not a.no_none_check,
                                allow_none=True if a.allow_none else None)
    max_budget = a.max_think_budget or defaults.budget
    if defaults.budget > max_budget:
        raise SystemExit(f"the default think budget {defaults.budget} is above --max-think-budget {max_budget}")
    tokenizer = load_tokenizer(a.model)
    name = a.name or os.path.basename(os.path.normpath(a.model))
    servers = None
    try:
        if a.backend == "vllm":
            from .backends.vllm import VLLMBackend
            urls = _urls(a.vllm_url)
            if a.launch_vllm:
                from .launch import VLLMServers
                servers = VLLMServers(a.model, a.gpus, a.vllm_base_port, log_dir=a.vllm_log_dir, python=a.vllm_python,
                                      max_model_len=a.max_model_len,
                                      gpu_memory_utilization=a.gpu_memory_utilization, quantization=a.quantization,
                                      mtp=not a.no_mtp, max_images=0 if a.no_images else a.max_images,
                                      max_pixels=a.max_pixels)
                logging.info("starting %d vLLM server(s), logs in %s; %s", len(a.gpus), a.vllm_log_dir,
                             servers.settings())
                servers.up(a.startup_timeout)
                urls = servers.urls
            if not urls:
                raise SystemExit("the vllm backend needs --vllm-url (running servers) or --launch-vllm")
            backend = VLLMBackend(urls, tokenizer, waves=a.waves, wave_max=a.wave_max, wave_gather=a.wave_gather,
                                  images=not a.no_images)
        elif a.backend == "transformers":
            from .backends.transformers import TransformersBackend
            backend = TransformersBackend(a.model, tokenizer, folder.bidirectional, a.device, a.dtype)
        elif a.backend == "llamacpp":
            from .backends.llamacpp import LlamaCppBackend
            backend = LlamaCppBackend(a.llamacpp_url, tokenizer, folder.bidirectional)
        else:
            from .backends.mlx import MLXBackend
            backend = MLXBackend(a.model, tokenizer, folder.bidirectional)
        images = {"max_images": 0 if a.no_images else a.max_images, "max_pixels": a.max_pixels}
        app = create_app(folder, backend, name, defaults, max_budget, a.max_model_len, options, images)
        scheme = "https" if tls else "http"
        web.run_app(app, host=a.host, port=a.port, access_log=None, ssl_context=tls,
                    print=lambda _: logging.info("%s serving on %s://%s:%d/v1/systemone", name, scheme, a.host, a.port))
    finally:
        if servers:
            servers.stop()
    return 0


def _public_options(a: argparse.Namespace) -> tuple:
    """(ServeOptions, TLS context or None) from the serving-publicly options; exits with the reason when one is
    unusable. Error messages name files and lines, never keys."""
    from .serving import (DECISION_PATHS, OPEN_PATHS, AccessLog, ApiKeys, RequestLog, ServeOptions, WatchedFile,
                          parse_models_document, tls_context)

    if bool(a.ssl_certfile) != bool(a.ssl_keyfile):
        raise SystemExit("--ssl-certfile and --ssl-keyfile go together")
    path = a.openrouter_models_path
    if not path.startswith("/") or path in DECISION_PATHS + OPEN_PATHS + ("/v1/models",):
        raise SystemExit(f"--openrouter-models-path {path!r} must start with / and not be a route of the API")
    try:
        options = ServeOptions(
            api_keys=ApiKeys(a.api_keys_file) if a.api_keys_file else None, max_inflight=a.max_inflight,
            models_document=WatchedFile(a.openrouter_models_file, parse_models_document, "OpenRouter models document")
            if a.openrouter_models_file else None, models_path=path,
            access_log=AccessLog(a.access_log) if a.access_log else None,
            request_log=RequestLog(a.request_log) if a.request_log else None)
        tls = tls_context(a.ssl_certfile, a.ssl_keyfile) if a.ssl_certfile else None
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot serve: {e}") from None
    return options, tls


def cmd_launch(a: argparse.Namespace) -> int:
    from .launch import VLLMServers, launch_variables

    servers = VLLMServers(a.model, a.gpus, a.base_port, a.host, a.log_dir, a.python, a.served_model_name,
                          a.max_model_len, a.gpu_memory_utilization, a.quantization, mtp=not a.no_mtp,
                          max_images=0 if a.no_images else a.max_images, max_pixels=a.max_pixels)
    print(servers.settings(), file=sys.stderr, flush=True)
    if a.dry_run:
        for gpu, (cmd, _) in zip(a.gpus, servers.plan()):
            variables = launch_variables(gpu, servers.bidirectional)
            print(" ".join([f"{k}={variables[k]}" for k in sorted(variables)] + [shlex.join(cmd)]))
        return 0
    servers.up(a.startup_timeout)
    print("vLLM servers ready: " + ",".join(servers.urls), flush=True)
    print(f"serve with: startlux-decide serve --model {a.model} --vllm-url {','.join(servers.urls)}", flush=True)
    try:
        while all(p.poll() is None for p in servers.procs):
            time.sleep(2)
        dead = [i for i, p in enumerate(servers.procs) if p.poll() is not None]
        print(f"vLLM server(s) {dead} exited; stopping the others", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        servers.stop()


def _print_table(rows: list[dict]) -> None:
    print(f"{'metric':<22}{'n':>6}{'mean':>12}{'median':>12}{'p80':>12}{'p95':>12}")
    for r in rows:
        print(f"{r['metric']:<22}{r['n']:>6}{r['mean']:>12.1f}{r['median']:>12.1f}{r['p80']:>12.1f}{r['p95']:>12.1f}")


def cmd_bench(a: argparse.Namespace) -> int:
    from .bench import METRICS, ROW_FIELDS, bench, read_jsonl, summarize, write_csv

    items = read_jsonl(a.requests)
    rows = asyncio.run(bench(a.url, items, a.concurrency, a.warmup, a.timeout, a.controls))
    summary = summarize(rows, METRICS)
    write_csv(a.out, summary, ("metric", "n", "mean", "median", "p80", "p95"))
    if a.rows:
        write_csv(a.rows, rows, ROW_FIELDS)
    failed = sum(r["status"] != "ok" for r in rows)
    print(f"{len(rows)} requests, {failed} failed, concurrency {a.concurrency}")
    _print_table(summary)
    return 1 if failed else 0


def cmd_eval(a: argparse.Namespace) -> int:
    from . import evaluate as E
    from .bench import read_jsonl

    if a.compare:
        out = E.compare(read_jsonl(a.compare[0]), read_jsonl(a.compare[1]), a.resamples)
    elif a.score:
        out = E.score(read_jsonl(a.score))
    else:
        if not a.data or not a.out:
            raise SystemExit("eval needs --data and --out (or --score / --compare)")
        records = asyncio.run(E.run(a.url, read_jsonl(a.data), a.concurrency, a.timeout, a.controls))
        E.write_jsonl(a.out, records)
        out = E.score(records)
    print(json.dumps(out, indent=2))
    return 0


def cmd_quantize(a: argparse.Namespace) -> int:
    from . import quantize as Q

    if a.method == "fp8":
        print(Q.fp8_advice(a.model))
        return 0
    if not a.out:
        raise SystemExit(f"quantize {a.method} needs --out")
    if a.method == "gguf":
        if not a.llama_cpp:
            raise SystemExit("quantize gguf needs --llama-cpp (a llama.cpp checkout with a build of llama-quantize)")
        cmds, problems = Q.gguf_plan(a.model, a.out, a.llama_cpp, a.type)
        after = (f"serve: llama-server -m {a.out}/model-{a.type}.gguf -ngl 99 -c 65536 --port 8081\n"
                 f"       startlux-decide serve --model {a.out} --backend llamacpp "
                 "--llamacpp-url http://127.0.0.1:8081")
    else:
        cmds, problems = Q.mlx_plan(a.model, a.out, a.bits)
        after = f"serve: startlux-decide serve --model {a.out} --backend mlx"
    for cmd in cmds:
        print(" ".join(cmd))
    print(f"then copy the tokenizer and config files of {a.model} that are missing in {a.out}")
    print(after)
    for problem in problems:
        print("missing: " + problem, file=sys.stderr)
    if a.run:
        if problems:
            return 1
        if a.method == "gguf":
            os.makedirs(a.out, exist_ok=True)
        Q.run(cmds)
        print("copied: " + ", ".join(Q.copy_folder_files(a.model, a.out)))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a = build_parser().parse_args(argv)
    commands = {"serve": cmd_serve, "launch-vllm": cmd_launch, "bench": cmd_bench, "eval": cmd_eval,
                "quantize": cmd_quantize}
    return commands[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
