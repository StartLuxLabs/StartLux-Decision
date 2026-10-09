"""Check that the fast kernels will be used for a model, before starting a server on it.

    python -m startlux_decision.check /path/to/StartLux-Decision-4B

Exits with status 1 when flash-linear-attention or causal-conv1d is missing or not importable; transformers would then
fall back to a plain torch path that is more than ten times slower.  Reports whether torch uses CUDA or ROCm.  On Apple
Silicon with mlx-lm installed the server runs the model with MLX instead, which needs neither.  Also says whether image
input will work (torch backend only).
"""
import importlib.util
import os
import sys

from .model import accelerator_name, fast_kernel_install_hint, fast_kernels_active


def images_ready(path):
    """"yes", or what image input still needs for the model in `path`."""
    if not os.path.exists(os.path.join(path, "preprocessor_config.json")):
        return "no, the model folder has no preprocessor_config.json"
    missing = [m for m in ("PIL", "torchvision") if importlib.util.find_spec(m) is None]
    return f"no, pip install {' '.join('pillow' if m == 'PIL' else m for m in missing)}" if missing else "yes"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip())
    from .server import mlx_available
    if mlx_available():                              # Apple Silicon: mlx-lm has its own kernels for these layers
        print("Apple Silicon: the server runs the model with MLX (text only); nothing else to install")
        sys.exit(0)
    import torch
    accelerator = accelerator_name()
    detail = ""
    if accelerator in ("cuda", "rocm"):
        runtime = torch.version.hip if accelerator == "rocm" else torch.version.cuda
        detail = f" ({torch.cuda.get_device_name()}, runtime {runtime})"
    print(f"accelerator: {accelerator}{detail}")
    ok = fast_kernels_active(sys.argv[1])
    print("fast kernels: " + ("active" if ok else f"NOT active, {fast_kernel_install_hint()}"))
    print("images: " + images_ready(sys.argv[1]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
