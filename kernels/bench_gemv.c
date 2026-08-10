/* Q4_0 GEMV 微基准：先验正确性，再测带宽利用率。
 *
 * 输出里最该看的不是 "多少 ms"，而是**达到了多少 GB/s**。
 * decode 是 memory-bound，一个 GEMV 内核跑到接近实测内存带宽就基本到头了，
 * 此时再优化指令序列没有意义 —— 该去动的是模型本身（更激进的量化、更小的模型）
 * 或者访存模式（权重布局、预取）。
 *
 * 默认矩阵尺寸取 Qwen2.5-1.5B 的一个 FFN 权重形状（8960 x 1536），
 * 这样测出来的数字对得上真实模型里的热点算子。
 */

/* -std=c11 会藏起 POSIX 的声明，clock_gettime 需要显式打开。
 * 必须在任何 include 之前定义。 */
#define _POSIX_C_SOURCE 200809L

#include "q4_gemv.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* xorshift32：要的是可复现，不是统计质量。 */
static uint32_t rng_state = 0x12345678u;
static uint32_t rnd(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}

static void fill_weights(block_q4_0 *w, size_t n_blocks) {
    for (size_t i = 0; i < n_blocks; i++) {
        w[i].d = fp32_to_fp16(0.01f + (float)(rnd() % 100) * 0.0001f);
        for (int j = 0; j < QK4_0 / 2; j++) w[i].qs[j] = (uint8_t)(rnd() & 0xFF);
    }
}

static void fill_acts(block_q8_0 *x, size_t n_blocks) {
    for (size_t i = 0; i < n_blocks; i++) {
        x[i].d = fp32_to_fp16(0.02f + (float)(rnd() % 100) * 0.0002f);
        for (int j = 0; j < QK8_0; j++) x[i].qs[j] = (int8_t)(rnd() & 0xFF);
    }
}

static int verify(const float *a, const float *b, int n, double tol) {
    double worst = 0.0;
    int worst_i = -1;
    for (int i = 0; i < n; i++) {
        const double denom = fabs(a[i]) > 1e-6 ? fabs(a[i]) : 1e-6;
        const double rel = fabs((double)a[i] - (double)b[i]) / denom;
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    if (worst > tol) {
        fprintf(stderr,
                "正确性检查失败：第 %d 行相对误差 %.3e（上限 %.1e），"
                "scalar=%.6f opt=%.6f\n",
                worst_i, worst, tol, a[worst_i], b[worst_i]);
        return 0;
    }
    printf("正确性检查通过（最大相对误差 %.3e）\n", worst);
    return 1;
}

typedef void (*gemv_fn)(int, int, const block_q4_0 *, const block_q8_0 *, float *);

static double bench(gemv_fn fn, int rows, int cols,
                    const block_q4_0 *w, const block_q8_0 *x, float *y,
                    int iters) {
    /* 预热：把权重拉进 cache 层级并让频率爬上来。
     * 注意权重通常远大于 L3（本例约 74 MiB），实测的就是 DRAM 带宽。 */
    fn(rows, cols, w, x, y);

    double best = 1e30;
    for (int i = 0; i < iters; i++) {
        const double t0 = now_sec();
        fn(rows, cols, w, x, y);
        const double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    int rows = 8960;   /* Qwen2.5-1.5B FFN intermediate size */
    int cols = 1536;   /* hidden size */
    int iters = 20;

    if (argc > 1) rows = atoi(argv[1]);
    if (argc > 2) cols = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);

    if (rows <= 0 || cols <= 0 || iters <= 0) {
        fprintf(stderr, "用法：%s [rows] [cols] [iters]，三个参数都要 > 0\n", argv[0]);
        return 2;
    }
    if (cols % QK4_0 != 0) {
        fprintf(stderr, "cols 必须是 %d 的倍数，收到 %d\n", QK4_0, cols);
        return 2;
    }

    const int nb = cols / QK4_0;
    const size_t n_wblocks = (size_t)rows * nb;

    block_q4_0 *w = malloc(n_wblocks * sizeof(block_q4_0));
    block_q8_0 *x = malloc((size_t)nb * sizeof(block_q8_0));
    float *y_ref = malloc((size_t)rows * sizeof(float));
    float *y_opt = malloc((size_t)rows * sizeof(float));
    if (!w || !x || !y_ref || !y_opt) {
        fprintf(stderr, "内存分配失败（需要约 %.1f MiB）\n",
                (double)(n_wblocks * sizeof(block_q4_0)) / (1 << 20));
        free(w); free(x); free(y_ref); free(y_opt);
        return 1;
    }

    fill_weights(w, n_wblocks);
    fill_acts(x, (size_t)nb);

    const double mib = (double)q4_gemv_bytes(rows, cols) / (1 << 20);
    printf("矩阵 %d x %d，权重 %.1f MiB，实现：%s\n\n",
           rows, cols, mib, q4_gemv_impl_name());

    q4_gemv_scalar(rows, cols, w, x, y_ref);
    q4_gemv_opt(rows, cols, w, x, y_opt);

    /* 两条路径的浮点累加顺序不同，逐位一致是不现实的；
     * 1e-4 的相对误差足以抓出真正的实现错误（比如 nibble 顺序搞反）。 */
    if (!verify(y_ref, y_opt, rows, 1e-4)) {
        free(w); free(x); free(y_ref); free(y_opt);
        return 1;
    }
    putchar('\n');

    const double bytes = (double)q4_gemv_bytes(rows, cols);
    const double ops = (double)q4_gemv_ops(rows, cols);

    const struct { const char *name; gemv_fn fn; } impls[] = {
        { "scalar", q4_gemv_scalar },
        { "opt",    q4_gemv_opt    },
    };

    double scalar_t = 0.0;
    printf("%-8s %10s %12s %12s %8s\n", "实现", "耗时(ms)", "带宽(GB/s)", "算力(GOPS)", "加速比");
    for (size_t i = 0; i < sizeof(impls) / sizeof(impls[0]); i++) {
        const double t = bench(impls[i].fn, rows, cols, w, x,
                               i == 0 ? y_ref : y_opt, iters);
        if (i == 0) scalar_t = t;
        printf("%-8s %10.3f %12.2f %12.2f %8.2fx\n",
               impls[i].name, t * 1e3, bytes / t / 1e9, ops / t / 1e9, scalar_t / t);
    }

    printf("\n判读方法：\n");
    printf("  带宽接近实测 DRAM 带宽 -> 已经 memory-bound，继续抠指令没用；\n");
    printf("  带宽远低于 DRAM 带宽   -> 还是 compute-bound 或访存模式差，有优化空间。\n");
    printf("  RK3588 的 LPDDR4x 标称 34.1 GB/s，实测一般在 12~20 GB/s，请以实测为准。\n");

    free(w); free(x); free(y_ref); free(y_opt);
    return 0;
}
