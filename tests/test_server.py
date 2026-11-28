"""The HTTP server on the fake engine: answers, details and timings, request ids, errors, Instant folders and many
requests in flight. Needs aiohttp."""
from __future__ import annotations

import asyncio
import shutil
import time
import unittest

from startlux_decide.backends.base import BackendError
from startlux_decide.backends.fake import FakeBackend
from startlux_decide.modelcfg import load_model_folder

from .folders import make_folder
from .test_pipeline import QUESTIONS, closing

try:
    from aiohttp.test_utils import TestClient, TestServer

    from startlux_decide.server import create_app
except ImportError:                       # the pure-logic suite runs without aiohttp
    TestClient = None


@unittest.skipIf(TestClient is None, "needs aiohttp")
class ServerTest(unittest.IsolatedAsyncioTestCase):
    kind = "auto_think"

    async def asyncSetUp(self):
        self.path = make_folder(self.kind)
        self.backend = self.make_backend()
        self.client = TestClient(TestServer(create_app(load_model_folder(self.path), self.backend, "test-model")))
        await self.client.start_server()

    def make_backend(self):
        return FakeBackend(script=closing)

    async def asyncTearDown(self):
        await self.client.close()
        shutil.rmtree(self.path, ignore_errors=True)

    async def post(self, body=None, **kw):
        r = await self.client.post("/v1/systemone", json=body, **kw)
        return r, await r.json()


class AnswerTest(ServerTest):
    async def test_answer(self):
        r, out = await self.post({"state": "charged twice", "questions": QUESTIONS, "model": "anything"})
        self.assertEqual(r.status, 200)
        self.assertEqual(set(out), {"answers", "usage", "model"})
        self.assertEqual(out["model"], "test-model")
        self.assertEqual(list(out["answers"]), ["team", "urgent", "severity"])
        self.assertEqual(out["answers"]["urgent"]["type"], "noul")
        self.assertEqual(set(out["usage"]), {"input_tokens", "output_tokens"})
        rid = r.headers["x-request-id"]
        self.assertRegex(rid, r"^[0-9a-f]{32}$")
        self.assertEqual(r.headers["x-typesafe-request-id"], rid)
        r, _ = await self.post({"state": "s", "questions": QUESTIONS}, headers={"x-request-id": "client-42"})
        self.assertEqual(r.headers["x-request-id"], "client-42")

    async def test_allow_none(self):
        r, out = await self.post({"state": "charged twice", "questions": {"team": QUESTIONS["team"]}})
        self.assertEqual(list(out["answers"]["team"]), ["type", "choice", "confidence", "probabilities"])
        r, out = await self.post({"state": "charged twice", "questions": {"team": QUESTIONS["team"]},
                                  "allow_none": True, "think_mode": "never"})
        self.assertEqual(r.status, 200)
        a = out["answers"]["team"]
        self.assertEqual(list(a), ["type", "choice", "confidence", "probabilities", "none_probability"])
        self.assertEqual(list(a["probabilities"]), ["billing", "shipping"])
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)
        self.assertTrue(a["choice"] is None or a["choice"] in a["probabilities"])
        self.assertTrue(0 <= a["none_probability"] <= 1)

    async def test_details_and_timings(self):
        r, out = await self.post({"state": "s", "questions": QUESTIONS, "think_mode": "always", "return_thought": True,
                                  "mix": 0.5, "think_budget": 64})
        self.assertEqual(r.status, 200)
        self.assertEqual(set(out["timing_ms"]), {"wall", "first_pass_sum", "thought_span", "after_thought_sum"})
        det = out["answers"]["team"]["details"]
        self.assertEqual((det["thought"]["ended"], det["mix"], det["gate"]), ("closed", 0.5, None))
        self.assertIn("Billing handles double charges.", det["thought"]["text"])
        self.assertTrue(all(c[2] == 64 for c in self.backend.calls if c[0] == "generate"))

    async def test_routes(self):
        r = await self.client.get("/v1/models")
        models = (await r.json())["models"]
        self.assertEqual([m["name"] for m in models], ["test-model"])
        self.assertRegex(models[0]["release_date"], r"^\d{4}-\d\d-\d\d$")
        r = await self.client.get("/health")
        h = await r.json()
        self.assertEqual((h["status"], h["kind"], h["defaults"]["think_mode"], h["defaults"]["gate"],
                          h["defaults"]["first_read"]), ("ok", self.kind, "auto", 0.6, "plain"))
        r = await self.client.get("/nope")
        self.assertEqual(r.status, 404)
        self.assertEqual((await r.json())["detail"][0]["type"], "http_error")
        self.assertIn("x-request-id", r.headers)
        r = await self.client.get("/v1/systemone")
        self.assertEqual(r.status, 405)


class ErrorTest(ServerTest):
    async def check(self, response, loc, fragment=""):
        r, out = response
        self.assertEqual(r.status, 422, out)
        self.assertEqual(out["detail"][0]["loc"], loc)
        self.assertIn(fragment, out["error"])
        self.assertIn("x-request-id", r.headers)

    async def test_validation(self):
        r = await self.client.post("/v1/systemone", data=b"{not json", headers={"Content-Type": "application/json"})
        self.assertEqual((r.status, (await r.json())["detail"][0]["type"]), (422, "json_invalid"))
        await self.check(await self.post([1, 2]), ["body"], "JSON object")
        await self.check(await self.post({"state": "s", "questions": {}}), ["body", "questions"], "non-empty")
        await self.check(await self.post({"state": "s", "questions": QUESTIONS, "gate": 2}), ["body", "gate"])
        await self.check(await self.post({"state": "s", "questions": QUESTIONS, "think_mode": "sometimes"}),
                         ["body", "think_mode"])
        await self.check(await self.post({"state": "s", "questions": QUESTIONS, "images": ["aGk="]}),
                         ["body", "images"], "text only")
        await self.check(await self.post({"state": "s", "questions": QUESTIONS, "allow_none": "yes"}),
                         ["body", "allow_none"], "true or false")
        await self.check(await self.post({"state": "s", "questions": {"team": dict(QUESTIONS["team"], allow_none=1)}}),
                         ["body", "questions", "team", "allow_none"], "true or false")
        bad = {"s": {"type": "score", "instructions": "q", "criteria": [str(i) for i in range(11)]}}
        await self.check(await self.post({"state": "s", "questions": bad}), ["body", "questions", "s"], "2 to 10")


class InstantServerTest(ServerTest):
    kind = "instant"

    async def test_instant(self):
        h = await (await self.client.get("/health")).json()
        self.assertEqual((h["kind"], h["prompt_attention"], h["defaults"]["think_mode"]), ("instant", "bidirectional",
                                                                                          "never"))
        r, out = await self.post({"state": "s", "questions": QUESTIONS, "return_details": True})
        self.assertEqual(r.status, 200)
        self.assertFalse(any(a["details"]["thought"]["used"] for a in out["answers"].values()))
        r, out = await self.post({"state": "s", "questions": QUESTIONS, "think_mode": "always"})
        self.assertEqual(r.status, 422)
        self.assertIn("Instant", out["error"])


class FailingEngineTest(ServerTest):
    def make_backend(self):
        class Broken(FakeBackend):
            async def readout(self, prompt, letter_ids):
                raise BackendError("server 1 is down")
        return Broken()

    async def test_502(self):
        with self.assertLogs("startlux_decide.server", "ERROR") as logs:
            r, out = await self.post({"state": "s", "questions": QUESTIONS})
        self.assertEqual(r.status, 502)
        self.assertNotIn("server 1 is down", out["error"])          # engine details stay in the server log
        self.assertIn(r.headers["x-request-id"], out["error"])
        self.assertIn("server 1 is down", logs.output[0])


class ClientsTest(ServerTest):
    async def test_bench_and_eval(self):
        from startlux_decide import bench, evaluate
        url = str(self.client.make_url("/")).rstrip("/")
        items = [{"id": f"r{i}", "state": f"ticket {i}", "questions": QUESTIONS,
                  "gold": {"team": "billing", "urgent": i % 2 == 0, "severity": 1}} for i in range(12)]
        rows = await bench.bench(url, items, concurrency=4, warmup=2, controls={"think_mode": "always"})
        self.assertEqual([r["status"] for r in rows], ["ok"] * 12)
        self.assertTrue(all(r["thought_questions"] == 3 and r["thought_span_ms"] > 0 for r in rows))
        summary = {r["metric"]: r for r in bench.summarize(rows)}
        self.assertEqual(summary["client_ms"]["n"], 12)
        a = await evaluate.run(url, items, concurrency=4, controls={"think_mode": "never"})
        b = await evaluate.run(url, items, concurrency=4, controls={"think_mode": "always"})
        m = evaluate.score(b)
        self.assertEqual((m["questions"], m["thought_share"]), (36, 100.0))
        c = evaluate.compare(a, b, resamples=200)
        self.assertEqual((c["paired_questions"], c["paired_requests"], c["thought_agreement"]), (36, 12, 0.0))


class ConcurrencyTest(ServerTest):
    def make_backend(self):
        return FakeBackend(script=closing, delay=0.25)

    async def test_many_requests_in_flight(self):
        t0 = time.perf_counter()
        results = await asyncio.gather(*(self.post({"state": f"s{i}", "questions": QUESTIONS}) for i in range(40)))
        self.assertTrue(all(r.status == 200 for r, _ in results))
        self.assertLess(time.perf_counter() - t0, 3.0)        # 40 requests of 3 questions, 0.25 s per engine call


if __name__ == "__main__":
    unittest.main()
