"""HTTP server with the TypeSafe /v1/systemone API (aiohttp, asynchronous: many requests in flight, every question
runs its own pipeline).

    POST /v1/systemone   {"state", "questions", optional "images" and controls (see controls.py)}
                         -> {"answers", "usage", "model"} (+ "timing_ms" with return_details)
    POST /v1/decisions   the same (OpenRouter's name for this API); a "model" field in the request is ignored
    GET  /v1/models      {"models": [{"name", "description", "release_date"}]}
    GET  /health         status and model kind (always open); engine and defaults too when no API keys are set or
                         the request carries a valid key
    GET  /openrouter/models (configurable)  the operator's OpenRouter provider models document, served as it is

Options for serving publicly (ServeOptions): API keys (every route except /health then needs "Authorization: Bearer
<key>", else 401), an in-flight limit for decision requests (answered at once with 429 and Retry-After: 1 when
reached, never queued), the OpenRouter models document, an access log without content, and a request log with
content (serving.RequestLog, off unless configured).

Every response carries an ``x-request-id`` header (the client's own when it sent a valid one) and the same value in
``x-typesafe-request-id``. Errors are {"error": message, "detail": [{"loc", "msg", "type"}]}: 401 without a valid key,
422 for a request the server refuses (bad JSON, bad fields, a question the model cannot answer), 404 / 405 for unknown
routes, 429 at the in-flight limit, 502 when the engine fails. Error messages and the server's own log never contain
keys or request content; request content goes only to the request log.
"""
from __future__ import annotations

import json
import logging
import re
import time
import traceback
import uuid
from dataclasses import replace
from typing import Callable

from aiohttp import web

from .backends.base import Backend, BackendError
from .controls import Controls, RequestError
from .modelcfg import ModelFolder
from .pipeline import Decider
from .serving import DECISION_PATHS, OPEN_PATHS, RequestLog, ServeOptions, client_address, key_id, utc_now

log = logging.getLogger("startlux_decide.server")
MAX_BODY = 64 * 1024 * 1024
MAX_UNPARSED = 65536                                           # characters of a body that is not JSON, in the request log
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_RequestKey = getattr(web, "RequestKey", None)                 # typed request storage in newer aiohttp versions
_RID = _RequestKey("request_id", str) if _RequestKey else "startlux_request_id"
_STATS = _RequestKey("stats", dict) if _RequestKey else "startlux_stats"
_BODY = _RequestKey("body", object) if _RequestKey else "startlux_body"                    # for the request log
_LOGGED = _RequestKey("logged_reply", dict) if _RequestKey else "startlux_logged_reply"     # the reply with thoughts


class _Inflight:
    """Decision requests being answered right now."""

    def __init__(self) -> None:
        self.count = 0


DECIDER = web.AppKey("decider", Decider)
FOLDER = web.AppKey("folder", ModelFolder)
NAME = web.AppKey("name", str)
OPTIONS = web.AppKey("options", ServeOptions)
INFLIGHT = web.AppKey("inflight", _Inflight)


def error(status: int, message: str, loc: list | None = None, kind: str = "value_error") -> web.Response:
    """A JSON error in the FastAPI shape, plus a top-level "error" message."""
    return web.json_response({"error": message, "detail": [{"loc": loc or ["body"], "msg": message, "type": kind}]},
                             status=status)


@web.middleware
async def _envelope(request: web.Request, handler: Callable) -> web.StreamResponse:
    """Request id on every response; HTTP errors and unexpected failures as JSON; the access log line."""
    t0 = time.perf_counter()
    rid = request.headers.get("x-request-id", "")
    rid = rid if _REQUEST_ID.fullmatch(rid) else uuid.uuid4().hex
    request[_RID] = rid
    try:
        response = await handler(request)
    except web.HTTPException as e:
        response = error(e.status, e.reason or "error", ["path"], "http_error")
    except Exception as e:                      # noqa: BLE001  (a 500 must still be JSON; the log gets no content)
        log.error("request %s failed with %s:\n%s", rid, type(e).__name__,
                  "".join(traceback.format_tb(e.__traceback__)))
        response = error(500, f"internal error ({type(e).__name__}); request id {rid}", ["server"], "internal_error")
    response.headers["x-request-id"] = rid
    response.headers["x-typesafe-request-id"] = rid
    options = request.app[OPTIONS]
    if options.access_log is not None:
        options.access_log.write(_access_record(request, response.status, t0))
    if options.request_log is not None and request.method == "POST" and request.path in DECISION_PATHS:
        options.request_log.write(_request_record(request, response, t0, options.request_log))
    return response


def _access_record(request: web.Request, status: int, t0: float) -> dict:
    """The access log line of a request: no state, questions, answers, keys, query strings or unknown paths."""
    stats = request.get(_STATS) or {}
    timing = stats.get("timing_ms") or {}
    resource = getattr(getattr(request.match_info, "route", None), "resource", None)
    return {"time": utc_now(), "request_id": request[_RID], "method": request.method,
            "route": resource.canonical if resource is not None else None, "status": status,
            "total_ms": round((time.perf_counter() - t0) * 1000, 3),
            "client_ip": client_address(request.remote, request.headers)["ip"],
            "key_id": key_id(request.headers.get("Authorization")), "questions": stats.get("questions"),
            "input_tokens": stats.get("input_tokens"), "thought_tokens": stats.get("thought_tokens"),
            "wall_ms": timing.get("wall"), "first_pass_sum_ms": timing.get("first_pass_sum"),
            "thought_span_ms": timing.get("thought_span"), "after_thought_sum_ms": timing.get("after_thought_sum")}


def _request_record(request: web.Request, response: web.StreamResponse, t0: float, content: RequestLog) -> dict:
    """The request log line of a decision request: who sent it with which key, what it sent (images as references to
    their stored files) and what came back (with details and thoughts, also when the request did not ask for them).
    A request refused before its body was read (401, 429) has "request": null."""
    body = request.get(_BODY)
    if isinstance(body, dict) and isinstance(body.get("images"), list):
        try:
            body = {**body, "images": [content.image(v) for v in body["images"]]}
        except OSError as e:
            body = {**body, "images": {"not_stored": type(e).__name__, "count": len(body["images"])}}
    reply = request.get(_LOGGED)
    if reply is None and isinstance(getattr(response, "body", None), (bytes, bytearray)):
        try:
            reply = json.loads(response.body)
        except ValueError:
            reply = None
    return {"time": utc_now(), "request_id": request[_RID], "route": request.path, "status": response.status,
            "total_ms": round((time.perf_counter() - t0) * 1000, 3),
            "client": client_address(request.remote, request.headers),
            "key_id": key_id(request.headers.get("Authorization")), "body_bytes": request.content_length,
            "request": body, "response": reply}


def _as_requested(answers: dict, details: bool, thought_text: bool) -> dict:
    """Answers computed with details and thoughts for the request log, cut back to what the request asked for."""
    out = {}
    for key, answer in answers.items():
        answer = dict(answer)
        info = answer.pop("details", None)
        if details and info is not None:
            if not thought_text and info.get("thought"):
                info = {**info, "thought": {k: v for k, v in info["thought"].items() if k != "text"}}
            answer["details"] = info
        out[key] = answer
    return out


@web.middleware
async def _guard(request: web.Request, handler: Callable) -> web.StreamResponse:
    """API keys for every route but /health, then the in-flight limit of decision requests (429 at once)."""
    options = request.app[OPTIONS]
    if options.api_keys is not None and request.path not in OPEN_PATHS \
            and not options.api_keys.check(request.headers.get("Authorization")):
        response = error(401, "missing or invalid API key: send the header Authorization: Bearer <key>",
                         ["header", "authorization"], "unauthorized")
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    if options.max_inflight is None or request.method != "POST" or request.path not in DECISION_PATHS:
        return await handler(request)
    inflight = request.app[INFLIGHT]
    if inflight.count >= options.max_inflight:
        response = error(429, f"the server is at its limit of {options.max_inflight} requests in flight; retry in a "
                              "second", ["server"], "too_many_requests")
        response.headers["Retry-After"] = "1"
        return response
    inflight.count += 1
    try:
        return await handler(request)
    finally:
        inflight.count -= 1


async def decisions(request: web.Request) -> web.Response:
    """POST /v1/systemone and /v1/decisions."""
    decider = request.app[DECIDER]
    keep = request.app[OPTIONS].request_log is not None
    try:
        body = await request.json()
    except json.JSONDecodeError as e:                       # the message has a position, no content
        if keep:
            request[_BODY] = {"unparsed": (await request.text())[:MAX_UNPARSED]}
        return error(422, f"the body is not valid JSON: {e}", ["body"], "json_invalid")
    except UnicodeDecodeError:
        return error(422, "the body is not UTF-8 text", ["body"], "json_invalid")
    request[_BODY] = body
    if not isinstance(body, dict):
        return error(422, "the body must be a JSON object", ["body"])
    questions = body.get("questions")
    request[_STATS] = {"questions": len(questions) if isinstance(questions, dict) else None}
    try:
        controls = decider.controls(body)
        run = replace(controls, details=True, thought_text=True) if keep else controls   # the log keeps thoughts
        decision = await decider.decide(body.get("state"), questions, run, images=body.get("images"))
    except RequestError as e:
        return error(422, str(e), e.loc)
    except BackendError as e:
        log.error("request %s: engine failure: %s", request[_RID], e)
        return error(502, f"the inference engine failed; request id {request[_RID]}", ["server"], "backend_error")
    request[_STATS].update({"input_tokens": decision.usage["input_tokens"],
                            "thought_tokens": decision.usage["output_tokens"], "timing_ms": decision.timing_ms})
    out = {"answers": decision.answers, "usage": decision.usage, "model": request.app[NAME]}
    if keep:
        request[_LOGGED] = {**out, "timing_ms": decision.timing_ms}
        out["answers"] = _as_requested(decision.answers, controls.details, controls.thought_text)
    if controls.details:
        out["timing_ms"] = decision.timing_ms
    return web.json_response(out)


async def models(request: web.Request) -> web.Response:
    folder = request.app[FOLDER]
    kind = {"instant": "Instant (one pass)", "v1": "v1 (one pass)"}.get(folder.kind, "AutoThink (thinks when unsure)")
    return web.json_response({"models": [{"name": request.app[NAME], "release_date": folder.release_date,
                                          "description": f"StartLux-Decision typed decision model, {kind}"}]})


async def openrouter_models(request: web.Request) -> web.Response:
    """The operator's OpenRouter provider models document, as it is in its file."""
    return web.Response(body=request.app[OPTIONS].models_document.get(), content_type="application/json")


async def health(request: web.Request) -> web.Response:
    decider = request.app[DECIDER]
    out = {"status": "ok", "model": request.app[NAME], "kind": decider.folder.kind}
    keys = request.app[OPTIONS].api_keys
    if keys is not None and not keys.check(request.headers.get("Authorization")):
        return web.json_response(out)                     # public: no engine addresses or settings
    d = decider.defaults
    out["images"] = decider.reads_images
    out.update({
        "prompt_attention": decider.folder.prompt_attention, "engine": decider.backend.describe(),
        "defaults": {"think_mode": d.think_mode, "gate": d.gate, "think_budget": d.budget, "mix": d.mix,
                     "thought_generation": decider.generation(d) if decider.can_think else None,
                     "first_read": d.first_read if decider.can_think else None},
        "max_think_budget": decider.max_budget, "context_length": decider.context_length})
    return web.json_response(out)


def create_app(folder: ModelFolder, backend: Backend, name: str, defaults: Controls | None = None,
               max_budget: int | None = None, context_length: int | None = None,
               options: ServeOptions | None = None, images: dict | None = None) -> web.Application:
    """The aiohttp application. The engine is started when the app starts and closed when it stops."""
    options = options or ServeOptions()
    app = web.Application(client_max_size=MAX_BODY, middlewares=[_envelope, _guard])
    app[FOLDER], app[NAME], app[OPTIONS], app[INFLIGHT] = folder, name, options, _Inflight()

    async def start(app: web.Application) -> None:
        await backend.start()
        app[DECIDER] = Decider(folder, backend, defaults, max_budget, context_length, **(images or {}))
        log.info("%s ready: %s model, engine %s", name, folder.kind, backend.describe())

    async def stop(app: web.Application) -> None:
        await backend.close()
        if options.access_log is not None:
            options.access_log.close()
        if options.request_log is not None:
            options.request_log.close()

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    for path in DECISION_PATHS:
        app.router.add_post(path, decisions)
    app.router.add_get("/v1/models", models)
    for path in OPEN_PATHS:
        app.router.add_get(path, health)
    if options.models_document is not None:
        app.router.add_get(options.models_path, openrouter_models)
    return app
