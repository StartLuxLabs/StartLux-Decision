"""Decision Index engine for StartLux-Decision, for the kit's own runner:

    python -m decision_index run --engine eval.di.startlux_decision_engine:StartLuxDecisionEngine --option model=/path/to/StartLux-Decision-4B \
        --out runs/StartLux-Decision-4B

Run it from the repository root with the kit installed.  One request at a time.  GPU graphs speed up single short
requests, but on the Decision Index mix, where many requests carry several longer questions, --option graphs=false is
faster.  For a full run use eval/di/batched.py instead, which writes the same result records and is faster again.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from decision_index.engines.base import Engine, Unsupported  # noqa: E402


class StartLuxDecisionEngine(Engine):
    name = "StartLux-Decision"
    latency = "In-process request wall time including prompt rendering and letter readout; excludes model loading."

    def __init__(self, model, graphs=True, **options):
        super().__init__(model=model, graphs=graphs, **options)
        from startlux_decision import StartLuxDecision
        if isinstance(graphs, str):
            graphs = graphs.lower() not in ("0", "false", "no", "off")
        self.m = StartLuxDecision(model, graphs=bool(graphs))
        self.provenance = {"kind": "startlux_decision", "model": model, "temperature": self.m.temperature}

    def runtime(self):
        return {"torch": torch.__version__, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "accelerator": self.m.accelerator, "fast_kernels": self.m.fast_kernels,
                "graphs": len(self.m.graphs), "cuda_graphs": len(self.m.graphs)}

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __call__(self, state, questions):
        try:
            answers, _ = self.m.decide(state, questions)
        except ValueError as e:
            if str(e).startswith("length "):
                raise Unsupported("maximum context length: " + str(e))
            raise
        return {"answers": answers}, None
