#!/usr/bin/env python
"""Stage 0 -- prove torch can run a kernel on the GPU, not merely see it.

`torch.cuda.is_available()` returned True with a wheel whose HIP runtime segfaulted at the
first kernel launch (ADR 0015), so the only meaningful check is to launch one. A segfault
kills this process before it can print anything, so the diagnosis is printed first.
"""
import glob
import re
import sys

import torch


def host_rocm_versions() -> list[str]:
    return sorted(
        re.sub(r".*/rocm-", "", p) for p in glob.glob("/opt/rocm-*") if re.search(r"rocm-\d", p)
    )


def main() -> int:
    if torch.version.hip is None:
        print("FATAL: torch is not a ROCm build (torch.version.hip is None).", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FATAL: torch cannot see the GPU. Check /dev/kfd permissions "
              "(groups render, video) and the host ROCm install.", file=sys.stderr)
        return 1

    wheel_hip = torch.version.hip.split("-")[0]
    hosts = host_rocm_versions()
    print(f">> torch {torch.__version__}, HIP {wheel_hip}, device {torch.cuda.get_device_name(0)}")
    print(f">> host ROCm trees: {', '.join(hosts) or 'none found'}")
    minor = ".".join(wheel_hip.split(".")[:2])
    if hosts and not any(h.startswith(minor) for h in hosts):
        print(f"WARNING: wheel HIP {wheel_hip} does not match any host ROCm ({', '.join(hosts)}). "
              "A mismatch can segfault at the first kernel; see ADR 0015.", file=sys.stderr)

    print(">> running a bf16 matmul (a segfault, exit 139, here means the wheel's HIP runtime "
          "does not match the host ROCm; see ADR 0015)")
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    value = (a @ a).float().sum().item()
    if value != value:  # NaN
        print("FATAL: matmul returned NaN.", file=sys.stderr)
        return 1
    print(">> GPU kernel ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
