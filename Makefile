CLANG := xcrun clang
COMMON := -O2 -fobjc-arc -framework Foundation -framework IOSurface -framework CoreML -framework Accelerate -ldl -lobjc
TRAINING_SRCS := training/train.m training/ane_mil_gen.m training/ane_runtime.m training/iosurface_io.m training/model.m
TEST_SHAPES_SRCS := tests/test_shapes.m training/iosurface_io.m
TEST_MIL_SRCS := tests/test_mil.m training/ane_mil_gen.m training/model.m
TEST_RUNTIME_PARSE_SRCS := tests/test_runtime_parse.m training/ane_runtime.m
PROBE_RUNTIME_SRCS := tests/probe_runtime.m training/ane_mil_gen.m training/ane_runtime.m training/model.m

all: build/train build/test_shapes build/test_mil build/test_runtime_parse build/probe_runtime

build:
	mkdir -p build

build/train: build $(TRAINING_SRCS)
	$(CLANG) $(COMMON) -o $@ $(TRAINING_SRCS)

build/test_shapes: build $(TEST_SHAPES_SRCS)
	$(CLANG) $(COMMON) -o $@ $(TEST_SHAPES_SRCS)

build/test_mil: build $(TEST_MIL_SRCS)
	$(CLANG) $(COMMON) -o $@ $(TEST_MIL_SRCS)

build/test_runtime_parse: build $(TEST_RUNTIME_PARSE_SRCS)
	$(CLANG) $(COMMON) -o $@ $(TEST_RUNTIME_PARSE_SRCS)

build/probe_runtime: build $(PROBE_RUNTIME_SRCS)
	$(CLANG) $(COMMON) -o $@ $(PROBE_RUNTIME_SRCS)

test: build/test_shapes build/test_mil build/test_runtime_parse
	./build/test_shapes
	./build/test_mil
	./build/test_runtime_parse

tune-residual: build/train
	zsh ./tools/tune_residual_grid.sh

baseline-repeat: build/train
	zsh ./tools/baseline_repeat.sh

phase3-protocol: build/train
	./tools/phase3_protocol.py --root . --out build/phase3_protocol.json

phase3-golden: build/train
	./tools/phase3_protocol.py --root . --preset golden --out build/phase3_protocol_golden.json

phase5-stability: build/train
	./tools/phase3_protocol.py --root . --preset stability --out build/phase3_protocol_stability.json

phase5-validate:
	./tools/validate_phase3_artifact.py --path build/phase3_protocol_stability.json

phase5-report:
	./tools/generate_phase4_report.py --protocol build/phase3_protocol_stability.json --baseline-protocol build/phase3_protocol_golden_post_promotion.json --benchmark build/ane_gpu_benchmark.json --out build/phase5_report_stability.md

phase6-determinism: build/train
	./tools/phase3_protocol.py --root . --preset stability --out build/phase6_protocol_determinism.json

phase6-validate:
	./tools/validate_phase3_artifact.py --path build/phase6_protocol_determinism.json

phase6-report:
	./tools/generate_phase4_report.py --protocol build/phase6_protocol_determinism.json --baseline-protocol build/phase3_protocol_golden_post_promotion.json --benchmark build/ane_gpu_benchmark.json --out build/phase6_report_determinism.md

phase6-consistency-gate:
	python3 ./tools/phase6_consistency_gate.py --glob "build/phase6_protocol_determinism_trim10_run*.json" --min-runs 4 --required-qualification-rate 1.0 --required-promotion-passes 3 --require-same-family --out build/phase6_consistency_gate.json

phase3-selftest:
	./tools/phase3_protocol.py --self-test

phase3-validate:
	./tools/validate_phase3_artifact.py --path build/phase3_protocol.json

ane-gpu-bench: build/train
	./tools/ane_gpu_benchmark.py --root . --mode both --out build/ane_gpu_benchmark.json

phase4-report:
	./tools/generate_phase4_report.py --protocol build/phase3_protocol.json --benchmark build/ane_gpu_benchmark.json --out build/phase4_report.md

compile-smoke: build/probe_runtime
	ANE_PROBE_CASE=stem ./build/probe_runtime

# Python training loop on MPS/CPU with real data (FineWeb-Edu)
PYTHON := .venv/bin/python

train-pt:
	$(PYTHON) training/train_pt.py

train-pt-resume:
	$(PYTHON) training/train_pt.py --resume

train-pt-smoke:
	$(PYTHON) training/train_pt.py --steps 200 --batch 4 --log-every 10 --save-every 100

clean:
	rm -rf build
