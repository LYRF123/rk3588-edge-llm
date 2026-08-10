# 测量结果

**目前为空 —— 还没有上板。**

## 目录约定

```
results/
├── bench-*.json          真机测量结果
├── sysinfo-*.txt         对应板子的环境信息
└── synthetic/            合成数据（mock backend），不是真机结果
    └── bench-*.json
```

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
