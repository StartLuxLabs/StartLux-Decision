"""Per-request controls: names, ranges, the budget cap, and Instant models."""
from __future__ import annotations

import json
import unittest

from startlux_decide.controls import Controls, RequestError, parse_controls

DEFAULTS = Controls(think_mode="auto", gate=0.6, budget=4096, mix=0.75)


def parse(body, can_think=True, max_budget=8192):
    return parse_controls(body, DEFAULTS, max_budget, can_think)


class ControlsTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        self.assertEqual(parse({}), DEFAULTS)
        body = {"think_mode": "hybrid", "gate": 1, "mix": 0, "think_budget": 100, "return_details": True}
        self.assertEqual(parse(body), Controls("auto", 1.0, 100, 0.0, True, False))
        self.assertEqual(parse({"think_mode": "plain"}).think_mode, "never")
        self.assertEqual(parse({"think_mode": "think"}).think_mode, "always")
        self.assertEqual(parse({"think_budget": 100000}).budget, 8192)          # capped by the server
        self.assertEqual(parse({"gate": None}), DEFAULTS)                        # null = not given
        self.assertEqual((DEFAULTS.first_read, parse({"first_read": "thinking"}).first_read), ("plain", "thinking"))

    def test_return_thought_implies_details(self):
        c = parse({"return_thought": True, "return_details": False})
        self.assertEqual((c.details, c.thought_text), (True, True))

    def test_bad_values(self):
        cases = [({"think_mode": "maybe"}, "think_mode"), ({"think_mode": 1}, "think_mode"), ({"gate": 1.5}, "gate"),
                 ({"gate": -0.1}, "gate"), ({"gate": "0.5"}, "gate"), ({"gate": True}, "gate"),
                 ({"gate": json.loads("NaN")}, "gate"), ({"mix": 2}, "mix"), ({"think_budget": 0}, "think_budget"),
                 ({"think_budget": 1.5}, "think_budget"), ({"think_budget": True}, "think_budget"),
                 ({"think_budget": "100"}, "think_budget"), ({"return_details": "yes"}, "return_details"),
                 ({"return_thought": 1}, "return_thought"),
                 ({"allow_none": "yes"}, "allow_none"), ({"allow_none": 1}, "allow_none"),
                 ({"first_read": "cached"}, "first_read"), ({"first_read": True}, "first_read")]
        for body, field in cases:
            with self.assertRaises(RequestError, msg=str(body)) as cm:
                parse(body)
            self.assertEqual(cm.exception.loc, ["body", field])

    def test_instant_models_do_not_think(self):
        for mode in ("auto", "always", "hybrid", "think"):
            with self.assertRaisesRegex(RequestError, "Instant"):
                parse({"think_mode": mode}, can_think=False)
        self.assertEqual(parse({"think_mode": "never", "gate": 0.5}, can_think=False).think_mode, "never")


if __name__ == "__main__":
    unittest.main()
