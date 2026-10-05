# ADR 0016 — llama.cpp built with HIP from a project-local toolchain

- **Status:** Accepted
- **Date:** 2026-10-05

## Context

Building llama.cpp for the GPU needs `hipcc`, the HIP headers, `rocm-llvm`, `hipblas` and
`rocsolver`. The host has the ROCm 7.1.1 runtime but none of these, and no passwordless
sudo. The first migration script only checked for `hipcc` on `PATH`, so it could not
succeed, and passed `-DGGML_HIPBLAS=ON`, which the pinned commit no longer uses.

The pip `rocm-sdk-core 7.14` ships a `hipcc`, but a binary built against it links the 7.14
runtime that segfaults on this host (ADR 0015).

Prototype on this host (2026-10-05): the 7.1.1 `.deb` packages from
`https://repo.radeon.com/rocm/apt/7.1.1` unpacked with `dpkg -x` and overlaid on a
symlink copy of `/opt/rocm-7.1.1` built llama.cpp `bdeb855` for gfx1102 in 4m32s;
`llama-server --list-devices` reported `ROCm0: AMD Radeon RX 7600 (8176 MiB)` with
`LD_LIBRARY_PATH` unset, and `/completion` generated with `-ngl 99 -fa on`.

## Decision

1. `scripts/00_setup_rocm.sh` downloads the missing packages for the host's ROCm version
   (names resolved from the repository's `Packages.gz`), extracts them under `vendor/rocm`,
   and builds `vendor/rocm/prefix`: a symlink tree of the host install with the extracted
   files overlaid. Caches are keyed by ROCm version (`Packages-<ver>-<dist>`,
   `debs/<ver>`, `extract-<ver>`, with `vendor/rocm/extract` a symlink to the current
   one), and download errors fail the script. It is idempotent and requires no root.
2. `scripts/00_setup_llamacpp.sh` builds with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1102`,
   `CMAKE_HIP_COMPILER` and `CMAKE_PREFIX_PATH` pointing at the prefix, using Ninja. The
   pinned commit is unchanged. It then fails the build unless `llama-server --list-devices`
   shows a ROCm device with `LD_LIBRARY_PATH` unset.
3. `scripts/00_setup_cuda.sh` is removed.

## Consequences

- ~4GB under `vendor/rocm` (not tracked by git): the downloaded packages plus the extracted
  tree.
- The toolchain version follows the host ROCm install; a host upgrade changes what is
  downloaded on the next `make setup`.
- Needs network access to `repo.radeon.com` the first time; the script stops with the URL
  when it is unreachable.
- Invariant 5 holds: base and tuned GGUFs go through the same binaries.
- Serving VRAM for the 8B Q4_K_M model on this card has not been measured; only the
  prototype generation above was run.

## Alternatives considered

- **Vulkan backend (`-DGGML_VULKAN=ON`).** Needs no ROCm, but `glslc` is also absent and the
  backend is slower on RDNA3. Not needed once HIP built.
- **The pip `hipcc` 7.14.** Links the runtime that crashes here.
- **Installing `hip-dev` with apt.** Needs sudo.
