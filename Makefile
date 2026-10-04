CONFIG ?= configs/qwen3-8b-personas.yaml
UV     ?= uv

# Variável essencial para RX 7600 (RDNA 3 / gfx1102)
# Exporta para todas as receitas do Makefile
export HSA_OVERRIDE_GFX_VERSION ?= 11.0.0

.PHONY: setup data train export serve eval report smoke test clean

setup:
	$(UV) sync --extra train --extra eval --extra dev
	# 2. Validação de detecção do ROCm no PyTorch
	$(UV) run python -c "import torch; assert torch.cuda.is_available(), 'GPU AMD não visível ao torch via ROCm'; print('PyTorch ROCm/HIP:', torch.version.hip, '| GPU:', torch.cuda.get_device_name(0))"
	# 3. Execução dos scripts adaptados para AMD
	@if [ -f scripts/00_setup_rocm.sh ]; then bash scripts/00_setup_rocm.sh; fi
	bash scripts/00_setup_llamacpp.sh

data:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG)

train:
	$(UV) run python scripts/02_train.py --config $(CONFIG)

export:
	$(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which tuned
	$(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which base

serve:
	bash scripts/04_serve.sh $(MODEL)

eval:
	$(UV) run python scripts/05_evaluate.py --config $(CONFIG)

report:
	$(UV) run python scripts/06_report.py --config $(CONFIG)

test:
	$(UV) run --extra dev pytest -v

smoke:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG) --smoke
	@echo "Smoke data ready. Run: make train export eval report"

clean:
	rm -rf outputs/merged-16bit-* outputs/gguf/*-f16.gguf