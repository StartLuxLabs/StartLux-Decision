"""The HTTP engines against fake servers over real HTTP: the vLLM client end to end against the release proxy (two
servers, waves on), streamed thoughts (server-sent events, closing a stream ends it at the server), start-up checks,
retries and refusals; the llama-server client. Needs aiohttp."""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import unittest
import unittest.mock
from types import SimpleNamespace

from startlux_decide.backends.base import BackendError, BackendRequestError, Sampling
from startlux_decide.backends.fake import FakeTokenizer
from startlux_decide.controls import Controls
from startlux_decide.pipeline import Decider
from startlux_decide.readout import probabilities

from . import reference_impl as ref
from .fake_vllm import FakeVLLM
from .test_golden import MODES, folder_for, package_room, random_question, random_settings, random_state

try:
    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from startlux_decide.backends.llamacpp import LlamaCppBackend
    from startlux_decide.backends.vllm import VLLMBackend
except ImportError:                       # the pure-logic suite runs without aiohttp
    aiohttp = None


class FakeServer:
    """A fake engine behind real HTTP. `handler(path, body or None)` returns (status, JSON or text); an async iterator
    of JSON events is sent as server-sent events ending in "data: [DONE]", `event_delay` seconds apart, and closed
    when the client goes away."""

    def __init__(self, handler, event_delay=0.0):
        self.handler, self.calls, self.event_delay = handler, 0, event_delay

    async def _serve(self, request):
        self.calls += 1
        body = await request.json() if request.method == "POST" else None
        status, out = await self.handler(str(request.url), body)
        if status != 200:
            return web.Response(status=status, text=out)
        if hasattr(out, "__aiter__"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            try:
                async for event in out:
                    await response.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                    await asyncio.sleep(self.event_delay)
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
            except (ConnectionError, RuntimeError):         # the client closed the connection
                pass
            finally:
                await out.aclose()                          # a stream left before its end counts as aborted
            return response
        return web.json_response(out)

    async def __aenter__(self):
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._serve)
        self.server = TestServer(app)
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")
        return self

    async def __aexit__(self, *exc):
        await self.server.close()


def vllm_handler(fake: FakeVLLM):
    async def handle(url, body):
        return 200, await fake(url, body)
    return handle


async def reference_complete(session, url, body):
    """The release proxy's complete(): four tries, 400/413/422 final."""
    for attempt in range(4):
        try:
            async with session.post(url, json=body) as r:
                if r.status == 200:
                    return await r.json()
                text = await r.text()
                if r.status in (400, 413, 422):
                    raise ValueError(text[:300])
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1 + attempt)
    raise RuntimeError("backend failed")


@unittest.skipIf(aiohttp is None, "needs aiohttp")
class VLLMOverHTTPTest(unittest.IsolatedAsyncioTestCase):
    async def test_against_release_proxy_with_waves(self):
        r = random.Random(99)
        cases = 0
        for case in range(60):
            s = random_settings(r)
            state = random_state(r)
            questions = {f"q{i}": random_question(r) for i in range(r.choice([1, 2, 3, 4]))}
            with self.subTest(case=case):
                await self.compare(s, state, questions)
                cases += 1
        self.assertEqual(cases, 60)

    async def compare(self, s, state, questions):
        tok = FakeTokenizer(open_think=s.open_think)
        ref_fakes = [FakeVLLM(s.max_len, drop_rate=s.drop) for _ in range(2)]
        new_fakes = [FakeVLLM(s.max_len, drop_rate=s.drop) for _ in range(2)]
        async with FakeServer(vllm_handler(ref_fakes[0])) as r0, FakeServer(vllm_handler(ref_fakes[1])) as r1, \
                FakeServer(vllm_handler(new_fakes[0])) as n0, FakeServer(vllm_handler(new_fakes[1])) as n1, \
                aiohttp.ClientSession() as session:
            a = SimpleNamespace(mode=s.mode, tau=s.tau, mix=s.mix, budget=s.budget, prefix=s.prefix,
                                tail_chars=s.tail_chars, rep_pen=1.0, stop_loops=0, loop_chunk=s.loop_chunk,
                                max_len=s.max_len, no_waves=False, wave_max=128, wave_gather=0.01)
            proxy = ref.Proxy(a, tok, s.temp, s.tthink, [r0.url + "/v1/completions", r1.url + "/v1/completions"],
                              lambda url, body: reference_complete(session, url, body))
            package_room(proxy, tok)
            expected = await proxy.handle(state, questions)
            for wave in proxy.waves.values():           # the release proxy never stopped its wave workers
                wave.task.cancel()
            await asyncio.gather(*(w.task for w in proxy.waves.values()), return_exceptions=True)
            backend = VLLMBackend([n0.url, n1.url], tok, wave_gather=0.01)
            await backend.start()
            n_probe = [len(f.log) for f in new_fakes]
            got = await Decider(folder_for(s), backend).decide(
                state, questions, Controls(think_mode=MODES[s.mode], gate=s.tau, budget=s.budget, mix=s.mix))
            await backend.close()
        self.assertEqual(sorted(ref_fakes[0].log + ref_fakes[1].log),
                         sorted(new_fakes[0].log[n_probe[0]:] + new_fakes[1].log[n_probe[1]:]))
        for key in questions:
            e, g = expected["answers"][key], got.answers[key]
            for field in ("choice", "noul", "score", "probabilities"):
                self.assertEqual(e.get(field), g.get(field), field)
        self.assertEqual(expected["usage"]["output_tokens"], got.usage["output_tokens"])

    async def test_start_checks(self):
        tok = FakeTokenizer()
        async with FakeServer(vllm_handler(FakeVLLM(max_model_len=32768))) as srv:
            backend = VLLMBackend([srv.url + "/v1/"], tok)
            await backend.start()
            self.assertEqual((backend.model_name, backend.context_length), ("m", 32768))
            self.assertEqual(backend.describe()["servers"], [srv.url])
            await backend.close()

        async def truncated(url, body):          # top_k applied to the read-out: letters missing
            out = await FakeVLLM()(url, body)
            if body and "logprobs" in body:
                top = out["choices"][0]["logprobs"]["top_logprobs"][0]
                for key in list(top)[20:]:
                    top[key] = -9999.0
            return 200, out
        async with FakeServer(truncated) as srv:
            backend = VLLMBackend([srv.url], tok)
            with self.assertRaisesRegex(BackendError, "--generation-config vllm"):
                await backend.start()
            await backend.close()

        async def few_logprobs(url, body):        # --max-logprobs below 26
            if body and body.get("logprobs", 0) > 20:
                return 400, "Requested sample logprobs of 26, which is greater than max allowed: 20"
            return 200, await FakeVLLM()(url, body)
        async with FakeServer(few_logprobs) as srv:
            backend = VLLMBackend([srv.url], tok)
            with self.assertRaisesRegex(BackendError, "max allowed: 20.*--max-logprobs 26"):
                await backend.start()
            await backend.close()

    async def test_streamed_thoughts(self):
        tok = FakeTokenizer()
        prompt = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nLet me think.\n"
        sampling = Sampling(0.6, 0.95, 20)
        fake = FakeVLLM(version="0.31.0", kinds=("close",), burst=3)
        async with FakeServer(vllm_handler(fake)) as srv:
            backend = VLLMBackend([srv.url], tok)
            await backend.start()
            self.assertEqual((backend.streams, backend.waves, backend.describe()["vllm_versions"]),
                             (True, False, ["0.31.0"]))
            events = [e async for e in backend.generate_stream(prompt, 4096, sampling, ["</think>"], 5)]
            whole = await backend.generate(prompt, 4096, sampling, ["</think>"], 5)
            self.assertEqual(("".join(e.text for e in events), events[-1].tokens, events[-1].finish_reason),
                             (whole.text, whole.tokens, whole.finish_reason))
            self.assertTrue(all(e.finish_reason is None for e in events[:-1]))
            self.assertEqual(fake.aborted, 0)
            await backend.close()

    async def test_closing_a_stream_ends_it_at_the_server(self):
        tok = FakeTokenizer()
        fake = FakeVLLM(version="0.31.0", kinds=("loop_short",))
        async with FakeServer(vllm_handler(fake), event_delay=0.001) as srv:
            backend = VLLMBackend([srv.url], tok)
            await backend.start()
            stream = backend.generate_stream("<|im_start|>assistant\n<think>\nP\n", 4096, Sampling(0.6, 0.95, 20),
                                             ["</think>"], 1)
            async for event in stream:
                if event.tokens >= 50:
                    break
            await stream.aclose()
            for _ in range(200):                            # the server notices at its next write
                if fake.aborted:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(fake.aborted, 1)
            await backend.close()

    async def test_stream_and_chunks_over_http(self):
        """Random requests through two servers: streamed thoughts give the answers of 1,024-token chunks."""
        r = random.Random(5)
        for case in range(25):
            s = random_settings(r)
            state = random_state(r)
            questions = {f"q{i}": random_question(r) for i in range(r.choice([1, 2, 3]))}
            tok = FakeTokenizer(open_think=s.open_think)
            got = {}
            for version in ("0.31.0", "0.19.1"):
                fakes = [FakeVLLM(s.max_len, drop_rate=s.drop, version=version) for _ in range(2)]
                async with FakeServer(vllm_handler(fakes[0])) as a, FakeServer(vllm_handler(fakes[1])) as b:
                    backend = VLLMBackend([a.url, b.url], tok, wave_gather=0.01)
                    await backend.start()
                    got[version] = await Decider(folder_for(s), backend).decide(
                        state, questions, Controls(think_mode=MODES[s.mode], gate=s.tau, budget=s.budget, mix=s.mix,
                                                   details=True, thought_text=True))
                    await backend.close()
            with self.subTest(case=case):
                for key in questions:
                    x, y = got["0.31.0"].answers[key], got["0.19.1"].answers[key]
                    self.assertEqual({k: v for k, v in x.items() if k != "details"},
                                     {k: v for k, v in y.items() if k != "details"})
                    tx, ty = x["details"]["thought"], y["details"]["thought"]
                    self.assertEqual([tx.get(k) for k in ("used", "tokens", "ended", "text")],
                                     [ty.get(k) for k in ("used", "tokens", "ended", "text")])

    async def test_retries_and_refusals(self):
        tok = FakeTokenizer()
        fake = FakeVLLM()
        state = {"fail": 2}

        async def flaky(url, body):
            if body and state["fail"] > 0:
                state["fail"] -= 1
                return 503, "busy"
            if body and "too long" in body["prompt"]:
                return 400, "This model's maximum context length is 65536 tokens"
            return 200, await fake(url, body)
        async with FakeServer(flaky) as srv:
            backend = VLLMBackend([srv.url], tok, retry_delay=0.0, model_name="m")
            await backend.start()                           # the probe goes through after two 503s
            self.assertEqual(srv.calls, 5)                  # models, version, three probes
            with self.assertRaisesRegex(BackendRequestError, "maximum context length"):
                await backend.readout("too long", [65, 66])
            self.assertEqual(srv.calls, 6)                  # a refusal is not tried again
            state["fail"] = 10
            with self.assertRaisesRegex(BackendError, "no answer after 4 attempts"):
                await backend.readout("x", [65, 66])
            await backend.close()


def llama_handler(post_sampling=True, bodies=None):
    async def handle(url, body):
        if body is None:
            return 200, {"default_generation_settings": {"n_ctx": 32768}}
        bodies.append(body)
        if body["n_predict"] == 1 and "logit_bias" in body:
            ids = [t for t, _ in body["logit_bias"]]
            r = random.Random(json.dumps(body["prompt"]))
            z = {t: r.uniform(-6, 3) for t in ids}
            if post_sampling:
                m = max(z.values())
                total = sum(math.exp(v - m) for v in z.values())
                top = sorted(({"id": t, "token": "?", "prob": math.exp(v - m) / total} for t, v in z.items()),
                             key=lambda c: -c["prob"])
                return 200, {"content": "A", "completion_probabilities": [{"id": top[0]["id"], "top_probs": top}]}
            return 200, {"content": "A", "completion_probabilities": [{"id": 5, "top_logprobs": [
                {"id": 5, "logprob": -0.1}, {"id": 6, "logprob": -2.5}]}]}
        n = min(body["n_predict"], 7)
        kind = {3: "word", 5: "eos"}.get(body["seed"] % 7, "limit")
        return 200, {"content": "x " * n, "tokens_predicted": n, "stop_type": kind}
    return handle


@unittest.skipIf(aiohttp is None, "needs aiohttp")
class LlamaCppTest(unittest.IsolatedAsyncioTestCase):
    async def test_readout_and_generation(self):
        tok, bodies = FakeTokenizer(), []
        async with FakeServer(llama_handler(True, bodies)) as srv:
            backend = LlamaCppBackend(srv.url, tok)
            await backend.start()
            self.assertEqual(backend.context_length, 32768)
            letters = list(range(65, 70))
            z = await backend.readout("Pick one.", letters)
            body = bodies[-1]
            self.assertEqual((body["n_probs"], body["top_k"], body["post_sampling_probs"], body["temperature"],
                              body["top_p"], body["min_p"], body["cache_prompt"]), (5, 5, True, 1.0, 1.0, 0.0, False))
            self.assertEqual(body["logit_bias"], [[t, 100.0] for t in letters])
            self.assertEqual(body["prompt"], tok.encode("Pick one."))
            r = random.Random(json.dumps(body["prompt"]))
            want = probabilities([r.uniform(-6, 3) for _ in letters], 1.0)
            for x, y in zip(probabilities(z, 1.0), want):
                self.assertAlmostEqual(x, y, places=12)
            finishes = set()
            for seed in range(7):
                g = await backend.generate("Think.", 100, Sampling(0.6, 0.95, 20), ["</think>"], seed)
                finishes.add((seed % 7, g.finish_reason))
                self.assertEqual((g.tokens, g.text), (7, "x " * 7))
            self.assertEqual(dict(finishes), {0: "length", 1: "length", 2: "length", 3: "stop", 4: "length",
                                              5: "stop", 6: "length"})
            body = bodies[-1]
            self.assertEqual((body["temperature"], body["top_p"], body["top_k"], body["min_p"], body["stop"],
                              body["n_predict"]), (0.6, 0.95, 20, 0.0, ["</think>"], 100))
            await backend.close()

    async def test_old_server_and_instant_folders(self):
        async with FakeServer(llama_handler(False, [])) as srv:
            backend = LlamaCppBackend(srv.url, FakeTokenizer())
            await backend.start()
            with self.assertRaisesRegex(BackendError, "post_sampling_probs"):
                await backend.readout("x", [65, 66])
            await backend.close()

    async def test_instant_folders(self):
        """Instant folders: the read-out asks the patched server for a bidirectional prompt and checks its echo
        (mode request), relies on a non-causal server context (mode server), or reads causally (mode off)."""
        def echoing(echo):
            base = llama_handler(True, bodies)
            async def handle(url, body):
                status, out = await base(url, body)
                if body is not None and "prompt_bidirectional" in body and echo:
                    out = dict(out, generation_settings={"prompt_bidirectional": body["prompt_bidirectional"]})
                return status, out
            return handle
        for mode, echo, ok in (("request", True, True), ("request", False, False), ("server", False, True),
                               ("off", False, True)):
            bodies = []
            with unittest.mock.patch.dict(os.environ, {"STARTLUX_LLAMACPP_BIDIR": mode}):
                async with FakeServer(echoing(echo)) as srv:
                    backend = LlamaCppBackend(srv.url, FakeTokenizer(), bidirectional=True)
                    await backend.start()
                    if ok:
                        p = await backend.readout("x", [65, 66])
                        self.assertAlmostEqual(sum(math.exp(v) for v in p), 1.0, places=6)
                    else:
                        with self.assertRaisesRegex(BackendError, "prompt_bidirectional"):
                            await backend.readout("x", [65, 66])
                    await backend.close()
            sent = [b for b in bodies if b.get("n_predict") == 1]
            self.assertEqual(all("prompt_bidirectional" in b for b in sent), mode == "request", mode)
        with unittest.mock.patch.dict(os.environ, {"STARTLUX_LLAMACPP_BIDIR": "maybe"}):
            with self.assertRaisesRegex(ValueError, "STARTLUX_LLAMACPP_BIDIR must be auto, request, server or off"):
                LlamaCppBackend("8081", FakeTokenizer(), bidirectional=True)

    async def test_bidir_auto(self):
        """STARTLUX_LLAMACPP_BIDIR=auto, the default, reads /props once at start: a server that reports patch 0003
        works as `server`, any other as `request`. Explicit values are not overridden, causal folders ignore it."""
        def handler(whole, bodies, props_calls):
            base = llama_handler(True, bodies)
            async def handle(url, body):
                status, out = await base(url, body)
                if body is None:
                    props_calls.append(url)
                    return status, (dict(out, mtmd_prompt_whole_when_non_causal=True) if whole else out)
                if "prompt_bidirectional" in body:                      # a server with patch 0001 echoes the field
                    out = dict(out, generation_settings={"prompt_bidirectional": body["prompt_bidirectional"]})
                return status, out
            return handle
        cases = ((None, True, "server"), (None, False, "request"), ("auto", True, "server"), ("auto", False, "request"),
                 ("", True, "server"), ("request", True, "request"), ("server", False, "server"), ("off", True, "off"))
        for setting, whole, want in cases:
            auto = setting in (None, "", "auto")
            bodies, props_calls = [], []
            with unittest.mock.patch.dict(os.environ):
                os.environ.pop("STARTLUX_LLAMACPP_BIDIR", None)
                if setting is not None:
                    os.environ["STARTLUX_LLAMACPP_BIDIR"] = setting
                async with FakeServer(handler(whole, bodies, props_calls)) as srv:
                    backend = LlamaCppBackend(srv.url, FakeTokenizer(), bidirectional=True)
                    self.assertEqual(backend.bidir_mode, "request" if auto else setting, "before start()")
                    await backend.start()
                    self.assertEqual(backend.bidir_mode, want, (setting, whole))
                    for _ in range(3):
                        p = await backend.readout("x", [65, 66])
                        self.assertAlmostEqual(sum(math.exp(v) for v in p), 1.0, places=6)
                    self.assertEqual(len(props_calls), 1, "/props is read once, at start")
                    sent = [b for b in bodies if b.get("n_predict") == 1]
                    self.assertEqual(["prompt_bidirectional" in b for b in sent], [want == "request"] * 3, (setting, whole))
                    d = backend.describe()
                    self.assertEqual(d["bidir_mode"], want + (" (auto)" if auto else ""))
                    self.assertEqual(d["prompt_attention"], {"request": "bidirectional (per request)",
                                                             "server": "bidirectional (server)",
                                                             "off": "causal (not the trained reading)"}[want])
                    await backend.close()
        # a causal (AutoThink) folder has no bidirectional mode: nothing is chosen, nothing is reported
        bodies, props_calls = [], []
        async with FakeServer(handler(True, bodies, props_calls)) as srv:
            backend = LlamaCppBackend(srv.url, FakeTokenizer())
            await backend.start()
            self.assertNotIn("bidir_mode", backend.describe())
            await backend.readout("x", [65, 66])
            self.assertFalse(any("prompt_bidirectional" in b for b in bodies))
            await backend.close()


if __name__ == "__main__":
    unittest.main()
