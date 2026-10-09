"""Fit one temperature per question type (choice, noul, score) on your own held-out decisions and write them into
decision_config.json.  Temperature never changes which option wins; it only sharpens or flattens the probabilities.

    python finetune/calibrate.py --model runs/mine/merged --dev dev.jsonl [--dry-run]

dev.jsonl uses the fine-tuning format (docs/finetuning.md).  Use data the model was not trained on, ideally from more
than one source, and check the result on a second held-out set: a temperature that fits one source can over- or
under-correct on another.  A type with fewer than 200 dev questions gets the pooled temperature.
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from startlux_decision import StartLuxDecision  # noqa: E402
from startlux_decision import jevfmt as J  # noqa: E402

GRID = [math.exp(math.log(0.2) + i * (math.log(5.0) - math.log(0.2)) / 800) for i in range(801)]


def nll(pairs, temp):
    total = 0.0
    for z, t in pairs:
        logp = torch.log_softmax(z / temp, -1)
        total -= float((t * logp).sum())
    return total / max(1, len(pairs))


def ece(pairs, temp, bins=10):
    conf, ok = [], []
    for z, t in pairs:
        p = torch.softmax(z / temp, -1)
        conf.append(float(p.max()))
        ok.append(float(int(p.argmax()) == int(t.argmax())))
    tot = 0.0
    for b in range(bins):
        sel = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins]
        if sel:
            tot += len(sel) / len(conf) * abs(sum(conf[i] for i in sel) / len(sel) - sum(ok[i] for i in sel) / len(sel))
    return round(tot, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--device", help="cuda (NVIDIA CUDA or AMD ROCm) or cpu (default: cuda when available)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    m = StartLuxDecision(a.model, device=a.device)
    by_type = {"choice": [], "noul": [], "score": []}
    with open(a.dev, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows, targets = [], []
            for key, q in rec["questions"].items():
                target = (rec.get("targets") or {}).get(key)
                row = J.from_systemone(rec["state"], q, qid=key)
                if target is None or (row["type"] == "choice" and len(row["options"]) > J.MAX_OPTIONS):
                    continue
                vec = torch.tensor([float(target.get(o["id"], 0.0)) for o in row["options"]])
                rows.append(row)
                targets.append(vec / vec.sum())
            if rows:
                logits, _ = m._logits(rows)
                for row, z, t in zip(rows, logits, targets):
                    by_type[row["type"]].append((z.float(), t))
    pooled = [x for v in by_type.values() for x in v]
    best = lambda pairs: min(GRID, key=lambda T: nll(pairs, T))
    t_pool = best(pooled)
    temps, report = {}, {}
    for typ, pairs in by_type.items():
        temps[typ] = best(pairs) if len(pairs) >= 200 else t_pool
        if pairs:
            report[typ] = {"n": len(pairs), "temperature": round(temps[typ], 4), "ece_before": ece(pairs, 1.0),
                           "ece_after": ece(pairs, temps[typ]), "nll_before": round(nll(pairs, 1.0), 4),
                           "nll_after": round(nll(pairs, temps[typ]), 4)}
    print(json.dumps({"pooled_temperature": round(t_pool, 4), "by_type": report}, indent=1))
    if not a.dry_run:
        path = os.path.join(a.model, "decision_config.json")
        cfg = json.load(open(path))
        cfg["temperature_by_type"] = {k: round(v, 4) for k, v in temps.items()}
        json.dump(cfg, open(path, "w"), indent=2)
        print("updated", path)


if __name__ == "__main__":
    main()
