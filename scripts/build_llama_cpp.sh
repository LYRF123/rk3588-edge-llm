#!/usr/bin/env bash
# 为 RK3588 编译 llama.cpp。
#
# 两种模式：
#   板上原生编译（推荐）—— 慢一点，但不用操心 sysroot 和 glibc 版本
#   x86 交叉编译     —— 快，但产物的 glibc 版本要跟板子对得上，否则跑不起来
#
# 用法：
#   bash scripts/build_llama_cpp.sh              # 板上原生编译
#   bash scripts/build_llama_cpp.sh --cross      # x86 交叉编译
#   LLAMA_CPP_DIR=/path/to/llama.cpp bash scripts/build_llama_cpp.sh
#
# 关于 -march：Cortex-A76 是 Armv8.2-A，有 dotprod 和 fp16，没有 i8mm 和 SVE。
# 给到 armv8.2-a+dotprod+fp16 能让 ggml 编进 dotprod 版的量化点积内核。
# 给更高的架构等级会在板子上 SIGILL；给 native 在交叉编译时无意义。

set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build-rk3588}"
JOBS="${JOBS:-$(nproc)}"
CROSS=0

for arg in "$@"; do
  case "$arg" in
    --cross) CROSS=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16 -mtune=cortex-a76"

if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  echo "在 $LLAMA_CPP_DIR 没找到 llama.cpp，先 clone："
  echo "  git clone https://github.com/ggml-org/llama.cpp $LLAMA_CPP_DIR"
  exit 1
fi

echo "==> llama.cpp 版本"
git -C "$LLAMA_CPP_DIR" log -1 --format='%H %cd %s' --date=short
echo "把这个 commit 记进 benchmark 结果里 —— llama.cpp 的性能在版本之间会明显变化，"
echo "不记 commit 的对比数据是没有意义的。"
echo

CMAKE_ARGS=(
  -S "$LLAMA_CPP_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_CURL=OFF          # 省掉 libcurl 依赖，板子上常常没有
  -DGGML_NATIVE=OFF         # 关掉自动探测，由我们显式指定 -march
)

if [ "$CROSS" = "1" ]; then
  command -v aarch64-linux-gnu-gcc >/dev/null || {
    echo "缺交叉编译器：apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu" >&2
    exit 1
  }
  echo "==> 交叉编译模式"
  CMAKE_ARGS+=(
    -DCMAKE_SYSTEM_NAME=Linux
    -DCMAKE_SYSTEM_PROCESSOR=aarch64
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
  )
else
  if [ "$(uname -m)" != "aarch64" ]; then
    echo "当前不是 aarch64，原生编译出来的东西在板子上跑不了。" >&2
    echo "要么在板子上跑这个脚本，要么加 --cross。" >&2
    exit 1
  fi
  echo "==> 板上原生编译"
fi

CMAKE_ARGS+=(
  -DCMAKE_C_FLAGS="$ARCH_FLAGS"
  -DCMAKE_CXX_FLAGS="$ARCH_FLAGS"
)

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

echo
echo "==> 产物"
ls -la "$BUILD_DIR/bin/" 2>/dev/null | grep -E 'llama-(cli|bench|server|quantize)' || \
  ls -la "$BUILD_DIR/bin/" 2>/dev/null | head -20

cat <<'EOF'

==> 下一步

1. 把 bin/ 加进 PATH，让 bench.backends.llama_cpp 能找到 llama-cli：
     export PATH="$BUILD_DIR/bin:$PATH"

2. 核对 dotprod 内核真的编进去了。跑一次 llama-bench，输出里应该能看到
   ggml 报告的 CPU 特性，确认 DOTPROD = 1：
     llama-bench -m <model.gguf> -p 64 -n 32

3. 绑大核再测一次，对比差别。RK3588 上让线程漂到 A55 小核会明显拖慢：
     taskset -c 4-7 llama-bench -m <model.gguf> -t 4 -p 64 -n 32
     taskset -c 0-7 llama-bench -m <model.gguf> -t 8 -p 64 -n 32
   预期是 4 大核 > 8 核混用（GEMM 同步，快核要等慢核），但**这是待验证的假设**，
   实测结果以板子为准。

EOF
