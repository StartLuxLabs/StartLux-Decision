import unittest
from unittest import mock

import torch

from startlux_decision.model import accelerator_name, fast_kernel_install_hint


class AcceleratorTest(unittest.TestCase):
    def test_rocm_uses_torch_cuda_device(self):
        with mock.patch.object(torch.version, "hip", "7.16"):
            self.assertEqual(accelerator_name("cuda"), "rocm")
            self.assertIn("--no-build-isolation", fast_kernel_install_hint("cuda"))

    def test_cuda_is_unchanged(self):
        with mock.patch.object(torch.version, "hip", None):
            self.assertEqual(accelerator_name("cuda"), "cuda")
            self.assertEqual(fast_kernel_install_hint("cuda"),
                             "pip install flash-linear-attention causal-conv1d")

    def test_non_cuda_device_is_unchanged(self):
        with mock.patch.object(torch.version, "hip", "7.16"):
            self.assertEqual(accelerator_name("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
