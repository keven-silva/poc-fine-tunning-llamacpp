CONFIG ?= configs/qwen3-0.6b-personas.yaml
UV     ?= uv

# The user's shell may export LD_LIBRARY_PATH entries for ROCm trees that do not exist
# (e.g. /opt/rocm-6.4.2). Every recipe that runs GPU code clears it so the torch wheel and
# the llama.cpp binaries resolve their own libraries (ADR 0015, 0016).
GPU_ENV = env -u LD_LIBRARY_PATH

.PHONY: setup data train export serve eval report smoke test clean

setup:
	$(UV) sync --extra train --extra eval --extra dev
	$(GPU_ENV) $(UV) run python scripts/00_check_gpu.py
	bash scripts/00_setup_rocm.sh
	bash scripts/00_setup_llamacpp.sh

data:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG)

train:
	$(GPU_ENV) $(UV) run python scripts/02_train.py --config $(CONFIG)

export:
	$(GPU_ENV) $(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which tuned
	$(GPU_ENV) $(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which base

serve:
	bash scripts/04_serve.sh $(MODEL)

eval:
	$(GPU_ENV) $(UV) run python scripts/05_evaluate.py --config $(CONFIG)

report:
	$(UV) run python scripts/06_report.py --config $(CONFIG)

test:
	$(UV) run --extra dev pytest -v

smoke:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG) --smoke
	@echo "Smoke data ready. Run: make train export eval report"

clean:
	rm -rf outputs/merged-16bit-* outputs/gguf/*-f16.gguf
