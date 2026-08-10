# Q4_0 GEMV 微基准。
#
#   make            本机编译（x86 上只有标量路径，用来验正确性）
#   make aarch64    交叉编译到 RK3588（需要 aarch64-linux-gnu-gcc）
#   make run        编译并跑
#   make clean
#
# A76 是 Armv8.2-A + dotprod + fp16，**没有** i8mm / SVE。
# -march 写成 armv8.2-a+dotprod+fp16 是刻意的：写 native 在交叉编译时无意义，
# 写更高的架构等级会在板子上直接 SIGILL。

CC      ?= cc
CFLAGS  ?= -O3 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

RK3588_ARCH := -march=armv8.2-a+dotprod+fp16 -mtune=cortex-a76
CROSS_CC    ?= aarch64-linux-gnu-gcc

SRC := q4_gemv.c bench_gemv.c
HDR := q4_gemv.h

.PHONY: all run aarch64 clean

all: bench_gemv

bench_gemv: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

# 本机就是 aarch64 时也用这条规则，-march 一样适用。
bench_gemv_rk3588: $(SRC) $(HDR)
	$(CROSS_CC) $(CFLAGS) $(RK3588_ARCH) -o $@ $(SRC) $(LDFLAGS)

aarch64: bench_gemv_rk3588
	@echo "产物：bench_gemv_rk3588 —— scp 到板子上跑"
	@file bench_gemv_rk3588 2>/dev/null || true

run: bench_gemv
	./bench_gemv

clean:
	rm -f bench_gemv bench_gemv_rk3588
