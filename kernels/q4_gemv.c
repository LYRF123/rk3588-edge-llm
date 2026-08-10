#include "q4_gemv.h"

#include <string.h>

/* A76 是 Armv8.2-A，有 ASIMD dotprod（SDOT/UDOT），没有 i8mm（Armv8.6）也没有 SVE。
 * 所以这里只走 vdotq_s32 这条路 —— 在 RK3588 上写 i8mm 的 SMMLA 内核是白写，
 * 编不过或者跑不了。编译时用 -march=armv8.2-a+dotprod+fp16 打开。 */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#  include <arm_neon.h>
#  define Q4_GEMV_HAVE_DOTPROD 1
#else
#  define Q4_GEMV_HAVE_DOTPROD 0
#endif

/* --- fp16 <-> fp32 -------------------------------------------------------
 * 走位运算而不是 __fp16，为的是让同一份代码在 x86 开发机上也能跑出**逐位一致**
 * 的结果。数值对不上的话，"NEON 版本比标量快 3 倍"这种结论就没法验证了。 */

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       /* +-0 */
        } else {
            /* 非规格化数：规格化成 fp32。前导零个数决定要左移多少。 */
            uint32_t m = mant;
            int shift = 0;
            while ((m & 0x400u) == 0) { m <<= 1; shift++; }
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - shift) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);   /* inf / nan */
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;

    if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);              /* 溢出成 inf */
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;           /* 下溢成 0 */
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        /* 就近舍入 */
        const uint32_t sub = (mant + (1u << (shift - 1))) >> shift;
        return (uint16_t)(sign | sub);
    }
    /* 就近舍入到偶数位 */
    const uint32_t rounded = mant + 0x0FFFu + ((mant >> 13) & 1u);
    if (rounded & 0x800000u) { exp += 1; mant = 0; }
    else                     { mant = rounded; }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* --- 标量参考 ------------------------------------------------------------ */

void q4_gemv_scalar(int n_rows, int n_cols,
                    const block_q4_0 *restrict w,
                    const block_q8_0 *restrict x,
                    float *restrict y) {
    const int nb = n_cols / QK4_0;

    for (int r = 0; r < n_rows; r++) {
        const block_q4_0 *wr = w + (size_t)r * nb;
        float acc = 0.0f;

        for (int b = 0; b < nb; b++) {
            int32_t sumi = 0;
            for (int j = 0; j < QK4_0 / 2; j++) {
                const int v0 = (int)(wr[b].qs[j] & 0x0F) - 8;
                const int v1 = (int)(wr[b].qs[j] >>   4) - 8;
                sumi += v0 * (int)x[b].qs[j];
                sumi += v1 * (int)x[b].qs[j + QK4_0 / 2];
            }
            acc += (float)sumi * fp16_to_fp32(wr[b].d) * fp16_to_fp32(x[b].d);
        }
        y[r] = acc;
    }
}

/* --- NEON dotprod -------------------------------------------------------- */

#if Q4_GEMV_HAVE_DOTPROD

/* 一次处理 4 行。
 *
 * 4 这个数字是这么来的：激活向量 x 在 4 行之间是复用的，一次加载喂 4 行能把
 * x 的加载开销摊掉 4 倍；同时 4 行需要 4 个累加寄存器 + 2 个 x 寄存器 +
 * 每行 2 个权重寄存器，寄存器压力还在 32 个 v 寄存器以内不会溢出。
 * 再往上（8 行）会 spill，A76 上实测通常反而变慢 —— 这一点上板后要用
 * bench_gemv 的 --rows-per-block 扫一遍确认，别照抄别人的结论。 */
static void q4_gemv_dotprod_4rows(int nb,
                                  const block_q4_0 *restrict w0,
                                  const block_q4_0 *restrict w1,
                                  const block_q4_0 *restrict w2,
                                  const block_q4_0 *restrict w3,
                                  const block_q8_0 *restrict x,
                                  float *restrict out /* [4] */) {
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t  s8b = vdupq_n_s8(8);

    float32x4_t acc = vdupq_n_f32(0.0f);
    const block_q4_0 *rows[4] = { w0, w1, w2, w3 };

    for (int b = 0; b < nb; b++) {
        /* 激活加载一次，4 行共用 —— 这是分块的全部意义所在 */
        const int8x16_t xl = vld1q_s8(x[b].qs);
        const int8x16_t xh = vld1q_s8(x[b].qs + QK4_0 / 2);
        const float xd = fp16_to_fp32(x[b].d);

        float partial[4];
        for (int k = 0; k < 4; k++) {
            const uint8x16_t q = vld1q_u8(rows[k][b].qs);

            /* 低 nibble -> 元素 0..15，高 nibble -> 元素 16..31，减 8 去偏移 */
            const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4b)), s8b);
            const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), s8b);

            /* 每条 SDOT 做 4 组 4-way 点积 = 16 次乘加。
             * 两条 SDOT 吃掉整个 32 元素的块。 */
            int32x4_t p = vdotq_s32(vdupq_n_s32(0), lo, xl);
            p = vdotq_s32(p, hi, xh);

            partial[k] = (float)vaddvq_s32(p) * fp16_to_fp32(rows[k][b].d) * xd;
        }
        acc = vaddq_f32(acc, vld1q_f32(partial));
    }
    vst1q_f32(out, acc);
}

void q4_gemv_opt(int n_rows, int n_cols,
                 const block_q4_0 *restrict w,
                 const block_q8_0 *restrict x,
                 float *restrict y) {
    const int nb = n_cols / QK4_0;
    int r = 0;

    for (; r + 3 < n_rows; r += 4) {
        q4_gemv_dotprod_4rows(nb,
                              w + (size_t)(r + 0) * nb,
                              w + (size_t)(r + 1) * nb,
                              w + (size_t)(r + 2) * nb,
                              w + (size_t)(r + 3) * nb,
                              x, y + r);
    }
    /* 尾部不足 4 行的走标量，n_rows 通常是 4 的倍数，这里只是兜底 */
    if (r < n_rows) {
        q4_gemv_scalar(n_rows - r, n_cols, w + (size_t)r * nb, x, y + r);
    }
}

const char *q4_gemv_impl_name(void) { return "neon-dotprod"; }

#else  /* 没有 dotprod：x86 开发机，或没开 -march=...+dotprod */

void q4_gemv_opt(int n_rows, int n_cols,
                 const block_q4_0 *restrict w,
                 const block_q8_0 *restrict x,
                 float *restrict y) {
    q4_gemv_scalar(n_rows, n_cols, w, x, y);
}

const char *q4_gemv_impl_name(void) { return "scalar (无 dotprod)"; }

#endif
