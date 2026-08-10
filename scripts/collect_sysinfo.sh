#!/usr/bin/env bash
# 采集板子的硬件与运行时信息，作为所有 benchmark 结果的附件。
#
# 上板第一件事就跑它。理由：RK3588 的板子之间差异很大 —— DDR 频率、
# 散热方案、内核版本、NPU 驱动版本都不一样，脱离这些信息的 tok/s 数字
# 没有可比性，也没法复现。
#
# 用法：bash scripts/collect_sysinfo.sh > results/sysinfo-$(hostname)-$(date +%F).txt

set -uo pipefail   # 刻意不加 -e：某一项读不到不该中断整份采集

section() { printf '\n===== %s =====\n' "$1"; }
try()     { echo "\$ $*"; "$@" 2>&1 || echo "  (失败或不可用)"; }
cat_if()  { if [ -r "$1" ]; then echo "--- $1"; cat "$1" 2>/dev/null; else echo "--- $1 (不可读)"; fi; }

section "基本信息"
try date -u
try hostname
try uname -a
cat_if /proc/device-tree/model
cat_if /etc/os-release

section "CPU"
try lscpu
echo "--- 各核当前频率 (kHz)"
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do
  [ -r "$f" ] && echo "$f = $(cat "$f")"
done
echo "--- 调频策略"
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
  [ -r "$f" ] && echo "$f = $(cat "$f")"
done
echo
echo "提示：RK3588 上 cpu0-3 是 A55 小核，cpu4-7 是 A76 大核。"
echo "跑 benchmark 前建议把 governor 设成 performance，否则测的是调频器的脾气："
echo "  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"

section "CPU 特性检查"
echo "关键：asimddp（dotprod）必须有；i8mm 和 sve 在 A76 上应当没有。"
echo "如果这里的结论和预期不符，kernels/ 里的 -march 就要改。"
if [ -r /proc/cpuinfo ]; then
  grep -m1 -i '^Features' /proc/cpuinfo || grep -m1 -i '^flags' /proc/cpuinfo
fi
for feat in asimddp i8mm sve fphp asimdhp; do
  if grep -qi "\b$feat\b" /proc/cpuinfo 2>/dev/null; then
    echo "  $feat: 有"
  else
    echo "  $feat: 无"
  fi
done

section "内存"
try free -h
echo "--- DDR 频率"
for f in /sys/class/devfreq/*ddr*/cur_freq /sys/kernel/debug/clk/clk_summary; do
  [ -r "$f" ] && echo "$f = $(head -c 200 "$f" 2>/dev/null)"
done
echo
echo "重要：标称带宽不等于可用带宽。roofline 要用实测值，用 mbw 或 STREAM 测："
echo "  mbw -n 10 512          # 简单粗暴，看 MEMCPY 那行"
echo "  测出来的数除以 34.1 就是 configs/models.yaml 里该填的 efficiency_bandwidth"

section "NPU"
for p in /sys/kernel/debug/rknpu/version /proc/rknpu/version; do cat_if "$p"; done
echo "--- devfreq 节点"
ls -d /sys/class/devfreq/*npu* 2>/dev/null || echo "  没找到 NPU devfreq 节点"
for d in /sys/class/devfreq/*npu*; do
  [ -d "$d" ] || continue
  for f in cur_freq available_frequencies governor; do
    [ -r "$d/$f" ] && echo "$d/$f = $(cat "$d/$f")"
  done
done
echo "--- 内核日志里的 rknpu"
dmesg 2>/dev/null | grep -i rknpu | tail -20 || echo "  (需要 root 才能读 dmesg)"

section "GPU (Mali-G610)"
ls -d /sys/class/devfreq/*gpu* /sys/class/misc/mali* 2>/dev/null || echo "  没找到 GPU 节点"
try clinfo -l

section "温度"
for z in /sys/class/thermal/thermal_zone*; do
  [ -r "$z/temp" ] || continue
  t=$(cat "$z/temp"); n=$(cat "$z/type" 2>/dev/null || basename "$z")
  echo "$n = $(awk "BEGIN{printf \"%.1f\", $t/1000}") C"
done
echo
echo "提示：上面是**空载**温度。真正要记录的是持续跑 10 分钟之后的温度和降频情况，"
echo "用 python -m bench run 里的 sustained-load 场景测。"

section "工具链"
try gcc --version
try cmake --version
try python3 --version

section "完"
echo "把这份输出连同 benchmark 结果一起归档。"
