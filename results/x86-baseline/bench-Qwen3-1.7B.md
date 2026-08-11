| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B Q4_K - Medium       |   1.19 GiB |     2.03 B | CPU        |       4 |           pp512 |        138.31 ± 1.88 |
| qwen3 1.7B Q4_K - Medium       |   1.19 GiB |     2.03 B | CPU        |       4 |           tg128 |         31.12 ± 0.02 |
| qwen3 1.7B Q4_K - Medium       |   1.19 GiB |     2.03 B | CPU        |       8 |           pp512 |        276.08 ± 0.12 |
| qwen3 1.7B Q4_K - Medium       |   1.19 GiB |     2.03 B | CPU        |       8 |           tg128 |         56.11 ± 0.91 |
| qwen3 1.7B Q4_K - Medium       |   1.19 GiB |     2.03 B | CPU        |      16 |           pp512 |        512.36 ± 1.48 |
| qwen3 1.7B Q4_K - Medium       |   1.19 GiB |     2.03 B | CPU        |      16 |           tg128 |         82.72 ± 0.76 |
| qwen3 1.7B Q8_0                |   2.01 GiB |     2.03 B | CPU        |       4 |           pp512 |        115.27 ± 0.02 |
| qwen3 1.7B Q8_0                |   2.01 GiB |     2.03 B | CPU        |       4 |           tg128 |         20.34 ± 0.01 |
| qwen3 1.7B Q8_0                |   2.01 GiB |     2.03 B | CPU        |       8 |           pp512 |        225.63 ± 0.03 |
| qwen3 1.7B Q8_0                |   2.01 GiB |     2.03 B | CPU        |       8 |           tg128 |         37.49 ± 0.01 |
| qwen3 1.7B Q8_0                |   2.01 GiB |     2.03 B | CPU        |      16 |           pp512 |       366.72 ± 15.79 |
| qwen3 1.7B Q8_0                |   2.01 GiB |     2.03 B | CPU        |      16 |           tg128 |         55.97 ± 1.12 |

build: 4c6766f (1)
