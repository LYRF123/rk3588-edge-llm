# 顶层入口。算子相关的目标在 kernels/Makefile 里。

PYTHON ?= python3

.PHONY: help test lint roofline mock kernel kernel-aarch64 sysinfo clean

help:
	@echo "不需要板子就能跑的："
	@echo "  make test            跑 Python 测试"
	@echo "  make lint            ruff 检查"
	@echo "  make roofline        理论上限分析"
	@echo "  make mock            合成 backend 跑通流水线"
	@echo "  make kernel          编译并跑算子微基准（x86 标量路径）"
	@echo "  make kernel-aarch64  交叉编译到 RK3588 并确认 SDOT 已生成"
	@echo ""
	@echo "需要板子的："
	@echo "  make sysinfo         采集板子信息（在板子上跑）"
	@echo "  bash scripts/build_llama_cpp.sh"

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	ruff check bench/ scripts/ tests/

roofline:
	$(PYTHON) -m bench roofline

mock:
	$(PYTHON) -m bench run --mock --out results
	@echo "生成报告：$(PYTHON) -m bench report results/synthetic/bench-*.json"

kernel:
	$(MAKE) -C kernels run

kernel-aarch64:
	$(MAKE) -C kernels aarch64
	@echo "SDOT 指令条数：$$(aarch64-linux-gnu-objdump -d kernels/bench_gemv_rk3588 | grep -c sdot)"

sysinfo:
	bash scripts/collect_sysinfo.sh

clean:
	$(MAKE) -C kernels clean
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
