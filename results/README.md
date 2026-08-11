# 测量结果

**RK3588 真机数据仍然为空 —— 还没有上板。**
目前只有 `x86-baseline/`：真实测量，但测的是 x86 服务器，不是端侧设备。

## 目录约定

```
results/
├── bench-*.json          RK3588 真机测量结果      ← 还没有
├── sysinfo-*.txt         对应板子的环境信息        ← 还没有
├── x86-baseline/         x86 服务器实测（真数据，非目标硬件）
│   ├── bench-*.md
│   └── sysinfo-x86.txt
└── synthetic/            合成数据（mock backend），不是真机结果
    └── bench-*.json
```

三类数据的区别必须一眼看得出来，因为它们的可信度完全不同：

| 目录 | 数据来源 | 能用来干什么 | 不能用来干什么 |
| --- | --- | --- | --- |
| 根目录 | RK3588 实测 | 一切 | — |
| `x86-baseline/` | x86 实测 | 验证工具链、验证量化产物没坏、看格式间的相对趋势 | **不能外推成 RK3588 的 tok/s** |
| `synthetic/` | mock backend 生成 | 验证框架代码路径 | 任何性能结论 |

`synthetic/` 的隔离是强制的：`bench.runner.run_suite` 在检测到合成 backend
却被要求写进非 `synthetic/` 目录时会直接抛 `ValueError`，
`tests/test_runner.py` 里有测试守着这条规矩。

理由：一份 benchmark 仓库里混进假数字，整个仓库就失去价值了。

## 每份结果都要配一份 sysinfo

RK3588 的板子之间差异很大——DDR 型号和频率、散热方案、内核版本、
NPU 驱动版本都不一样。脱离这些信息的 tok/s 数字没有可比性，也没法复现。

```bash
bash scripts/collect_sysinfo.sh > results/sysinfo-$(hostname)-$(date +%F).txt
```

## 生成报告

```bash
python3 -m bench report results/bench-20260810T093209Z.json -o results/report.md
```
