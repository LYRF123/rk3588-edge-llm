| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       4 |           pp512 |         58.26 ± 2.75 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       4 |           tg128 |         11.81 ± 0.38 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       8 |           pp512 |        113.33 ± 4.95 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       8 |           tg128 |         22.23 ± 0.27 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |      16 |           pp512 |        204.53 ± 0.66 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |      16 |           tg128 |         34.66 ± 0.27 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | CPU        |       4 |           pp512 |         45.27 ± 0.01 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | CPU        |       4 |           tg128 |          8.81 ± 0.00 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | CPU        |       8 |           pp512 |         74.90 ± 0.60 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | CPU        |       8 |           tg128 |         15.37 ± 0.36 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | CPU        |      16 |           pp512 |       157.84 ± 12.97 |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | CPU        |      16 |           tg128 |         25.90 ± 0.02 |

build: 4c6766f (1)
