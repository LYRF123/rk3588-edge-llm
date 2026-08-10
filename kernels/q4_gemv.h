/* Q4_0 x Q8_0 GEMV —— decode 阶段真正吃掉时间的那个算子。
 *
 * 为什么是 GEMV 而不是 GEMM：decode 每次只前向 1 个 token，所有权重矩阵乘法
 * 都退化成"矩阵 x 向量"。这类算子的算术强度约 2 ops/byte，彻底 memory-bound
 * —— 优化的目标不是省指令，而是**别浪费带宽**，同时让访存流水不被计算打断。
 *
 * 数据格式沿用 ggml 的 Q4_0 / Q8_0 分块量化，这样写出来的结论能直接对应到
 * llama.cpp 的实际表现，而不是一个自造格式上的玩具数字。
 *
 *   block_q4_0: 32 个权重 -> 1 个 fp16 scale + 16 字节 nibble = 18 字节
 *               有效位宽 18*8/32 = 4.5 bit/weight
 *   block_q8_0: 32 个激活 -> 1 个 fp16 scale + 32 字节 int8 = 34 字节
 *
 * nibble 的排布跟 ggml 一致：qs[j] 的低 4 位是第 j 个权重，高 4 位是第 j+16 个。
 * 反量化值 = (nibble - 8) * d。这个 -8 的偏移是 Q4_0 无零点对称量化的定义。
 */

#ifndef Q4_GEMV_H
#define Q4_GEMV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QK4_0 32
#define QK8_0 32

typedef struct {
    uint16_t d;                /* fp16 scale，按原始位存，用 fp16_to_fp32 解 */
    uint8_t  qs[QK4_0 / 2];    /* 16 字节，打包 32 个 4-bit 权重 */
} block_q4_0;

typedef struct {
    uint16_t d;                /* fp16 scale */
    int8_t   qs[QK8_0];        /* 32 个 int8 激活 */
} block_q8_0;

/* 编译期确认结构体没有被填充字节撑大 —— 多 2 个字节就是多 11% 的带宽。 */
typedef char q4_gemv_size_check[
    (sizeof(block_q4_0) == 18 && sizeof(block_q8_0) == 34) ? 1 : -1];

/* fp16 -> fp32，纯位运算，不依赖 _Float16 或 F16C。 */
float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);

/* 标量参考实现。作为正确性基准，不做任何优化。 */
void q4_gemv_scalar(int n_rows, int n_cols,
                    const block_q4_0 *restrict w,   /* [n_rows][n_cols/32] 行主序 */
                    const block_q8_0 *restrict x,   /* [n_cols/32] */
                    float *restrict y);             /* [n_rows] */

/* 优化实现。在支持 dotprod 的 ARM 上走 SDOT，否则退化为标量。
 * 用 q4_gemv_impl_name() 确认实际编到了哪条路径。 */
void q4_gemv_opt(int n_rows, int n_cols,
                 const block_q4_0 *restrict w,
                 const block_q8_0 *restrict x,
                 float *restrict y);

/* 返回 q4_gemv_opt 实际编译到的实现名，例如 "neon-dotprod" / "scalar"。 */
const char *q4_gemv_impl_name(void);

/* 每次 GEMV 需要从内存读的字节数（权重为主，激活可忽略但仍计入）。 */
static inline size_t q4_gemv_bytes(int n_rows, int n_cols) {
    const size_t nb = (size_t)n_cols / QK4_0;
    return (size_t)n_rows * nb * sizeof(block_q4_0) + nb * sizeof(block_q8_0);
}

/* 每次 GEMV 的乘加操作数（1 次乘 + 1 次加 = 2 ops）。 */
static inline size_t q4_gemv_ops(int n_rows, int n_cols) {
    return (size_t)n_rows * (size_t)n_cols * 2;
}

#ifdef __cplusplus
}
#endif

#endif /* Q4_GEMV_H */
