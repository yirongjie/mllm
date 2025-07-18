#include "GemmPack.hpp"
#include "Types.hpp"
#include <assert.h>
#include <cstdlib>
#include <float.h>
#include <math.h>
#include <stdio.h>  // for assert
#include <stdlib.h> // for qsort
#include <string.h>
#include "ComputeUtils.hpp"

int mllm_cpu_has_sve(void) {
#if defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int mllm_cpu_has_matmul_int8(void) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of
// blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in
// blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants
// bytes
//                         from bias offset form to pure sign form (this saves
//                         subtract operations durin unpacking)
//
static block_q4_0x4 make_block_q4_0x4(block_q4_0 *in, unsigned int blck_size_interleave,
                                      unsigned int xor_mask) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) { out.d[i] = in[i].d; }

    for (int i = 0; i < QK4_0 * 2; i++) {
        int src_offset = (i / (4 * blck_size_interleave)) * blck_size_interleave;
        int src_id = (i % (4 * blck_size_interleave)) / blck_size_interleave;
        src_offset += (i % blck_size_interleave);

        out.qs[i] = in[src_id].qs[src_offset] ^ xor_mask;
    }

    return out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of
// blck_size_interleave
static block_q4_0x8 make_block_q4_0x8(block_q4_0 *in, unsigned int blck_size_interleave,
                                      unsigned int xor_mask) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) { out.d[i] = in[i].d; }

    for (int i = 0; i < QK4_0 * 4; i++) {
        int src_offset = (i / (8 * blck_size_interleave)) * blck_size_interleave;
        int src_id = (i % (8 * blck_size_interleave)) / blck_size_interleave;
        src_offset += (i % blck_size_interleave);

        out.qs[i] = in[src_id].qs[src_offset] ^ xor_mask;
    }

    return out;
}

void quantize_q8_0_4x4(const float *__restrict x, void *__restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++)
                srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = MLLM_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 3] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 11] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 15] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0F; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = (d != 0.0F) ? 1.0F / d : 0.0F;

            y[i].d[row_iter] = MLLM_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

void quantize_q8_0_4x8(const float *__restrict x, void *__restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++)
                srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = MLLM_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 4; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][2 * j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][2 * j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][2 * j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0F; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = (d != 0.0F) ? 1.0F / d : 0.0F;

            y[i].d[row_iter] = MLLM_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

void quantize_mat_q8_0(const float *__restrict x, void *__restrict vy, int64_t nrow,
                       int64_t n_per_row, int64_t blck_size_interleave) {
    assert(nrow == 4);
    (void)nrow;
    if (blck_size_interleave == 4) {
        quantize_q8_0_4x4(x, vy, n_per_row);
    } else if (blck_size_interleave == 8) {
        quantize_q8_0_4x8(x, vy, n_per_row);
    } else {
        assert(false);
    }
}

static size_t quantize_q4_0_nr_bl(const float *__restrict src, void *__restrict dst, int64_t nrow,
                                  int64_t n_per_row, int nrows_interleaved,
                                  int blck_size_interleave) {
    assert(n_per_row % QK4_0 == 0);
    const int nb = n_per_row / QK4_0;

    void *out_ptr = NULL;
    if (nrows_interleaved == 8) {
        out_ptr = (block_q4_0x8 *)dst;
    } else if (nrows_interleaved == 4) {
        out_ptr = (block_q4_0x4 *)dst;
    }
    assert(nrows_interleaved <= 8);
    block_q4_0 dst_tmp[8];

    for (int b = 0; b < (nrow * n_per_row); b += nrows_interleaved * n_per_row) {
        for (int64_t x = 0; x < nb; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                quantize_row_q4_0(src + b + i * n_per_row + x * QK4_0, (block_q4_0 *)dst_tmp + i,
                                  QK4_0);
            }

            if (nrows_interleaved == 8) {
                *(block_q4_0x8 *)out_ptr = make_block_q4_0x8(dst_tmp, blck_size_interleave, 0x88);
                out_ptr = (block_q4_0x8 *)out_ptr + 1;
            } else if (nrows_interleaved == 4) {
                *(block_q4_0x4 *)out_ptr = make_block_q4_0x4(dst_tmp, blck_size_interleave, 0x88);
                out_ptr = (block_q4_0x4 *)out_ptr + 1;
            }
        }
    }

    return ((nrow * n_per_row) / QK4_0 * sizeof(block_q4_0));
}

size_t quantize_q4_0_4x4(const float *__restrict src, void *__restrict dst, int64_t nrows,
                         int64_t n_per_row, const float *imatrix) {
    if (imatrix != nullptr) { return quantize_q4_0_nr_bl(src, dst, nrows, n_per_row, 4, 4); }
    assert(false);
    return 0;
}

size_t quantize_q4_0_4x8(const float *__restrict src, void *__restrict dst, int64_t nrows,
                         int64_t n_per_row, const float *imatrix) {
    if (imatrix == nullptr) { return quantize_q4_0_nr_bl(src, dst, nrows, n_per_row, 4, 8); }
    assert(false);
    return 0;
}

size_t quantize_q4_0_8x8(const float *__restrict src, void *__restrict dst, int64_t nrows,
                         int64_t n_per_row, const float *imatrix) {
    if (imatrix == nullptr) { return quantize_q4_0_nr_bl(src, dst, nrows, n_per_row, 8, 8); }
    assert(false);
    return 0;
}

void gemv_q4_0_4x4_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                        const void *__restrict vy, int nr, int nc,
                        const void *__restrict bias) {
    if (bias != nullptr) {
        _gemv_q4_0_4x4_q8_0_bias(n, s, bs, vx, vy, nr, nc, bias);
        return;
    }

    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

// #if defined(__ARM_FEATURE_SVE)
//     if (svcntw() == 8) {
//         assert(!(mllm_cpu_has_sve() && (svcntw() == 8))
//                && "__ARM_FEATURE_SVE defined, use the Q4_0_8_8 quantization format "
//                   "for optimal performance");
//     }
// #endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const block_q8_0 *a_ptr_base = (const block_q8_0 *)vy;
    const block_q4_0x4 *b_ptr_base = (const block_q4_0x4 *)vx;

    for (int i_nc = 0; i_nc < nc; i_nc += ncols_interleaved) {
        float *s_ptr = s + i_nc;
        const block_q4_0x4 *b_ptr = b_ptr_base + (i_nc / ncols_interleaved) * nb;
        const block_q8_0 *a_ptr = a_ptr_base;
        __asm__ __volatile__(
            "movi v16.4s, #0\n" // v16 = acc0 = [0, 0, 0, 0] (32-bit accumulators for 4 outputs)
            "movi v17.4s, #0\n" // v17 = acc1
            "mov x5, %x[nb]\n"  // x5 = nb (block counter)
            "1:\n"              // Main loop over blocks (nb)
            // --- 加载 Scales ---
            "ldrh w6, [%x[a_ptr], #0]\n" // Load d_a (fp16 scale)
            "ldr s18, [%x[b_ptr], #0]\n" // Load d_b0, d_b1
            "ldr s19, [%x[b_ptr], #4]\n" // Load d_b2, d_b3
            "fmov s20, w6\n"             // Move d_a to float register
            "fcvtl v18.2s, v18.2h\n"     // Convert d_b0, d_b1 to fp32
            "fcvtl v19.2s, v19.2h\n"     // Convert d_b2, d_b3 to fp32
            "fcvtl v20.2s, v20.2h\n"     // Convert d_a to fp32
            "dup v18.4s, v18.s[0]\n"     // Broadcast d_b0
            "dup v19.4s, v19.s[0]\n"     // Broadcast d_b2
            "dup v20.4s, v20.s[0]\n"     // Broadcast d_a
            // --- Q8向量数据加载 (a) ---
            "ldr q0, [%x[a_ptr], #2]\n"  // Load first 16 bytes of a->qs
            "ldr q1, [%x[a_ptr], #18]\n" // Load second 16 bytes of a->qs
            // --- Q4权重数据加载 (b) ---
            "ldr q2, [%x[b_ptr], #8]\n"  // Load first 16 bytes of b->qs
            "ldr q3, [%x[b_ptr], #24]\n" // Load second 16 bytes of b->qs
            // --- 解包 Q4.0 权重到 Q8.0 ---
            // Unpack first 32x 4-bit quants into 32x 8-bit quants
            "movi v21.16b, #0x0f\n"         // low nibble mask
            "movi v22.16b, #-8\n"           // subtraction value
            "and v4.16b, v2.16b, v21.16b\n" // low nibbles
            "ushr v5.16b, v2.16b, #4\n"     // high nibbles
            "add v4.16b, v4.16b, v22.16b\n" // v4 = unpacked b quants 0..15
            "add v5.16b, v5.16b, v22.16b\n" // v5 = unpacked b quants 16..31
            // Unpack second 32x 4-bit quants
            "and v6.16b, v3.16b, v21.16b\n"
            "ushr v7.16b, v3.16b, #4\n"
            "add v6.16b, v6.16b, v22.16b\n" // v6
            "add v7.16b, v7.16b, v22.16b\n" // v7
            // --- 执行 4x4 矩阵乘法 ---
            // smmla acc, w, v
            // The 4x4 matrix is formed by the 16 bytes in the register.
            // We are doing a row-vector * matrix multiplication. The vector 'a' needs
            // to be treated as rows of a matrix.
            "smmla v16.4s, v4.16b, v0.16b\n" // acc0 += mat(v4) * mat(v0)
            "smmla v17.4s, v5.16b, v0.16b\n" // acc1 += mat(v5) * mat(v0)
            "smmla v16.4s, v6.16b, v1.16b\n"
            "smmla v17.4s, v7.16b, v1.16b\n"
            // --- 累加和转换 ---
            "add v16.4s, v16.4s, v17.4s\n"      // v16 has the final int32 sums for this block
            "scvtf v17.4s, v16.4s\n"            // Convert int32 sums to float32
            "fmul v18.4s, v18.4s, v20.4s\n"     // multiply scales d_a * d_b
            "fmla %v[sum].4s, v17.4s, v18.4s\n" // FMLA into final sum register
            // --- 循环控制 ---
            "add %x[a_ptr], %x[a_ptr], #34\n" // sizeof(block_q8_0) = 2+32
            "add %x[b_ptr], %x[b_ptr], #72\n" // sizeof(block_q4_0x4) = 8+64
            "subs x5, x5, #1\n"
            "bne 1b\n"
            // --- 存储结果 ---
            "str q[sum], [%x[s_ptr]]\n"
            : [sum] "+w"(s_ptr) // using "+w" for NEON registers
            : [a_ptr] "r"(a_ptr), [b_ptr] "r"(b_ptr), [nb] "r"(nb), [s_ptr] "r"(s_ptr)
            : "cc", "memory", "x5", "x6",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22");
    }
// AArch64 NEON (包括 Apple Silicon) 使用点积内联函数
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    float *res_ptr = s;

    __asm__ __volatile__("movi v31.16b, #0x4\n"
                         "movi v30.16b, #0xf0\n"
                         "add %x[b_ptr], %x[b_ptr], #0x8\n"
                         "1:" // Column loop
                         "add x22, %x[a_ptr], #0x2\n"
                         "movi v29.16b, #0x0\n"
                         "mov x21, %x[nb]\n"
                         "2:" // Block loop
                         "ldr q28, [%x[b_ptr], #0x0]\n"
                         "ldr q27, [x22, #0x0]\n"
                         "movi v26.4s, #0x0\n"
                         "sub x20, x22, #0x2\n"
                         "ldr q25, [x22, #0x10]\n"
                         "ldr q24, [%x[b_ptr], #0x10]\n"
                         "sub x21, x21, #0x1\n"
                         "add x22, x22, #0x22\n"
                         "ldr q23, [%x[b_ptr], #0x20]\n"
                         "ldr q22, [%x[b_ptr], #0x30]\n"
                         "ld1r { v21.8h }, [x20]\n"
                         "ldr q20, [%x[b_ptr], #-0x8]\n"
                         "sshl v16.16b, v28.16b, v31.16b\n"
                         "and v28.16b, v28.16b, v30.16b\n"
                         "sshl v19.16b, v24.16b, v31.16b\n"
                         "and v24.16b, v24.16b, v30.16b\n"
                         "add %x[b_ptr], %x[b_ptr], #0x48\n"
                         "sshl v18.16b, v23.16b, v31.16b\n"
                         "and v23.16b, v23.16b, v30.16b\n"
                         ".inst 0x4f9be21a  // sdot v26.4s, v16.16b, v27.4b[0]\n"
                         "sshl v17.16b, v22.16b, v31.16b\n"
                         "and v22.16b, v22.16b, v30.16b\n"
                         "fcvtl v21.4s, v21.4h\n"
                         "fcvtl v16.4s, v20.4h\n"
                         ".inst 0x4f99e39a  // sdot v26.4s, v28.16b, v25.4b[0]\n"
                         "fmul v16.4s, v16.4s, v21.4s\n"
                         ".inst 0x4fbbe27a  // sdot v26.4s, v19.16b, v27.4b[1]\n"
                         ".inst 0x4fb9e31a  // sdot v26.4s, v24.16b, v25.4b[1]\n"
                         ".inst 0x4f9bea5a  // sdot v26.4s, v18.16b, v27.4b[2]\n"
                         ".inst 0x4f99eafa  // sdot v26.4s, v23.16b, v25.4b[2]\n"
                         ".inst 0x4fbbea3a  // sdot v26.4s, v17.16b, v27.4b[3]\n"
                         ".inst 0x4fb9eada  // sdot v26.4s, v22.16b, v25.4b[3]\n"
                         "scvtf v26.4s, v26.4s, #0x4\n"
                         "fmla v29.4s, v26.4s, v16.4s\n"
                         "cbnz x21, 2b\n"
                         "sub %x[nc], %x[nc], #0x4\n"
                         "str q29, [%x[res_ptr], #0x0]\n"
                         "add %x[res_ptr], %x[res_ptr], #0x10\n"
                         "cbnz %x[nc], 1b\n"
                         : [b_ptr] "+&r"(b_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
                         : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
                         : "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                           "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22");
// x86 AVX-VNNI 和 AVX2 实现
#elif defined(__AVX2__)
#define CALC_SUMI_PARTIAL(V_B_Q4, V_A0_S16, V_A1_S16)                      \
    ({                                                                     \
        /* v0 = (int8_t)(L<<4), v1 = (int8_t)(H<<4) */                     \
        const __m128i low_nib_mask = _mm_set1_epi8(0x0F);                  \
        const __m128i v_L_nibbles = _mm_and_si128((V_B_Q4), low_nib_mask); \
        const __m128i v0s_u8 = _mm_slli_epi16(v_L_nibbles, 4);             \
        const __m128i v1s_u8 = _mm_andnot_si128(low_nib_mask, (V_B_Q4));   \
                                                                           \
        /* 符号扩展到16位 */                                               \
        const __m128i v0s_s16 = _mm_cvtepi8_epi16(v0s_u8);                 \
        const __m128i v1s_s16 = _mm_cvtepi8_epi16(v1s_u8);                 \
                                                                           \
        /* 核心计算: (v0*a0 + v1*a1) >> 4 */                               \
        const __m128i prod0 = _mm_mullo_epi16(v0s_s16, (V_A0_S16));        \
        const __m128i prod1 = _mm_mullo_epi16(v1s_s16, (V_A1_S16));        \
        const __m128i sum_prods_s16 = _mm_add_epi16(prod0, prod1);         \
        const __m128i terms_s16 = _mm_srai_epi16(sum_prods_s16, 4);        \
                                                                           \
        /* 水平求和 */                                                     \
        const __m128i ones = _mm_set1_epi16(1);                            \
        const __m128i sums_s32 = _mm_madd_epi16(terms_s16, ones);          \
        _mm_extract_epi32(sums_s32, 0) + _mm_extract_epi32(sums_s32, 1);   \
    })

    const block_q8_0 *a_ptr_base = (const block_q8_0 *)vy;
    const block_q4_0x4 *b_ptr_base = (const block_q4_0x4 *)vx;

    // 外层循环：处理不同的4列输出组
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 *b_ptr = b_ptr_base + x * nb;
        const block_q8_0 *a_ptr = a_ptr_base;

        float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // 中层循环：处理数据块
        for (int l = 0; l < nb; l++) {
            int32_t sumi_cols[4] = {0, 0, 0, 0};

            // 内层循环：处理块内的子区域 (k)
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                // 加载 a 向量的两个部分并符号扩展到16位
                const __m128i v_a0_s16 = _mm_cvtepi8_epi16(_mm_loadu_si32(a_ptr[l].qs + k * blocklen));
                const __m128i v_a1_s16 = _mm_cvtepi8_epi16(_mm_loadu_si32(a_ptr[l].qs + k * blocklen + qk / 2));

                // 加载 b 权重的4x4字节区域
                const __m128i v_b_block = _mm_loadu_si128((const __m128i *)(b_ptr[l].qs + k * 16));

                // ---- 循环展开: 手动处理 j = 0, 1, 2, 3 ----

                // j = 0
                sumi_cols[0] += CALC_SUMI_PARTIAL(v_b_block, v_a0_s16, v_a1_s16);

                // j = 1
                sumi_cols[1] += CALC_SUMI_PARTIAL(_mm_srli_si128(v_b_block, 4), v_a0_s16, v_a1_s16);

                // j = 2
                sumi_cols[2] += CALC_SUMI_PARTIAL(_mm_srli_si128(v_b_block, 8), v_a0_s16, v_a1_s16);

                // j = 3
                sumi_cols[3] += CALC_SUMI_PARTIAL(_mm_srli_si128(v_b_block, 12), v_a0_s16, v_a1_s16);
            }

            // --- 应用缩放因子并累加到浮点和 ---
            const __m128i sumi_vec = _mm_loadu_si128((const __m128i *)sumi_cols);
            const __m128 sumi_f = _mm_cvtepi32_ps(sumi_vec);

            const float d_a = MLLM_FP16_TO_FP32(a_ptr[l].d);
            const __m128 d_b = _mm_cvtph_ps(_mm_loadu_si128((const __m128i *)b_ptr[l].d));
            const __m128 scales = _mm_mul_ps(_mm_set1_ps(d_a), d_b);

            __m128 current_sumf = _mm_loadu_ps(sumf);
            current_sumf = _mm_add_ps(current_sumf, _mm_mul_ps(sumi_f, scales));
            _mm_storeu_ps(sumf, current_sumf);
        }

        // 存储最终结果
        _mm_storeu_ps(s + x * ncols_interleaved, _mm_loadu_ps(sumf));
    }
    // 确保宏只在当前代码块生效
#undef CALC_SUMI_PARTIAL
#else
    float sumf[4];
    int sumi;

    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     << 4);
                        const int v1 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i])
                                 + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]))
                                >> 4;
                    }
                    sumf[j] +=
                        sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j]) * MLLM_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
#endif
}

void _gemv_q4_0_4x4_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc,
                              const void *__restrict bias) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;
    (void)bias;

// #if defined(__ARM_FEATURE_SVE)
//     if (svcntw() == 8) {
//         assert(!(mllm_cpu_has_sve() && (svcntw() == 8))
//                && "__ARM_FEATURE_SVE defined, use the Q4_0_8_8 quantization format "
//                   "for optimal performance");
//     }
// #endif
// #if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
//     assert(!(mllm_cpu_has_neon() && mllm_cpu_has_matmul_int8())
//            && "__ARM_NEON and __ARM_FEATURE_MATMUL_INT8 defined, use the Q4_0_4_8 "
//               "quantization format for optimal performance");
// #elif defined(__ARM_NEON) && defined(__aarch64__)
#if defined(__ARM_NEON) && defined(__aarch64__)
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    const void *bias_ptr = bias;
    float *res_ptr = s;

    __asm__ __volatile__(
        "movi v31.16b, #0x4\n"
        "movi v30.16b, #0xf0\n"
        "add %x[b_ptr], %x[b_ptr], #0x8\n"
        "1:" // Column loop
        "add x22, %x[a_ptr], #0x2\n"
        "movi v29.16b, #0x0\n"
        "mov x21, %x[nb]\n"
        "2:" // Block loop
        "ldr q28, [%x[b_ptr], #0x0]\n"
        "ldr q27, [x22, #0x0]\n"
        "movi v26.4s, #0x0\n"
        "sub x20, x22, #0x2\n"
        "ldr q25, [x22, #0x10]\n"
        "ldr q24, [%x[b_ptr], #0x10]\n"
        "sub x21, x21, #0x1\n"
        "add x22, x22, #0x22\n"
        "ldr q23, [%x[b_ptr], #0x20]\n"
        "ldr q22, [%x[b_ptr], #0x30]\n"
        "ld1r { v21.8h }, [x20]\n"
        "ldr q20, [%x[b_ptr], #-0x8]\n"
        "sshl v16.16b, v28.16b, v31.16b\n"
        "and v28.16b, v28.16b, v30.16b\n"
        "sshl v19.16b, v24.16b, v31.16b\n"
        "and v24.16b, v24.16b, v30.16b\n"
        "add %x[b_ptr], %x[b_ptr], #0x48\n"
        "sshl v18.16b, v23.16b, v31.16b\n"
        "and v23.16b, v23.16b, v30.16b\n"
        ".inst 0x4f9be21a  // sdot v26.4s, v16.16b, v27.4b[0]\n"
        "sshl v17.16b, v22.16b, v31.16b\n"
        "and v22.16b, v22.16b, v30.16b\n"
        "fcvtl v21.4s, v21.4h\n"
        "fcvtl v16.4s, v20.4h\n"
        ".inst 0x4f99e39a  // sdot v26.4s, v28.16b, v25.4b[0]\n"
        "fmul v16.4s, v16.4s, v21.4s\n"
        ".inst 0x4fbbe27a  // sdot v26.4s, v19.16b, v27.4b[1]\n"
        ".inst 0x4fb9e31a  // sdot v26.4s, v24.16b, v25.4b[1]\n"
        ".inst 0x4f9bea5a  // sdot v26.4s, v18.16b, v27.4b[2]\n"
        ".inst 0x4f99eafa  // sdot v26.4s, v23.16b, v25.4b[2]\n"
        ".inst 0x4fbbea3a  // sdot v26.4s, v17.16b, v27.4b[3]\n"
        ".inst 0x4fb9eada  // sdot v26.4s, v22.16b, v25.4b[3]\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "fmla v29.4s, v26.4s, v16.4s\n"
        "cbnz x21, 2b\n"
        "sub %x[nc], %x[nc], #0x4\n"
        // -- bias start
        "ldr q28, [%x[bias_ptr], #0x0]\n"         // load data from bias_ptr
        "fadd v29.4s, v29.4s, v28.4s\n"           // add q29 = q29 + q28
        "add %x[bias_ptr], %x[bias_ptr], #0x10\n" // move bias_ptr to next side.
        // -- bias end
        "str q29, [%x[res_ptr], #0x0]\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "cbnz %x[nc], 1b\n"
        : [b_ptr] "+&r"(b_ptr), [bias_ptr] "+&r"(bias_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
        : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
        : "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22");
#else
    float sumf[4];
    int sumi;

    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    const float *bias_ptr = (const float *)bias;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

        // init reduction tmp sumf with bias.
        // the bias is not quanted and default precision is fp32.
        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = bias_ptr[x * ncols_interleaved + j];
        }

        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     << 4);
                        const int v1 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i])
                                 + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]))
                                >> 4;
                    }
                    sumf[j] +=
                        sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j]) * MLLM_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
#endif
}

void gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                        const void *__restrict vy, int nr, int nc,
                        const void *__restrict bias) {
    if (bias != nullptr) {
        _gemv_q4_0_4x8_q8_0_bias(n, s, bs, vx, vy, nr, nc, bias);
        return;
    }

    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

// #if defined(__ARM_FEATURE_SVE)
//     if (svcntw() == 8) {
//         assert(!(mllm_cpu_has_sve() && (svcntw() == 8))
//                && "__ARM_FEATURE_SVE defined, use the Q4_0_8_8 quantization format "
//                   "for optimal performance");
//     }
// #endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    float *res_ptr = s;

    __asm__ __volatile__("movi v2.16b, #0x4\n"
                         "movi v1.16b, #0xf0\n"
                         "add %x[b_ptr], %x[b_ptr], #0x8\n"
                         "1:" // Column loop
                         "add x23, %x[a_ptr], #0x2\n"
                         "movi v0.16b, #0x0\n"
                         "mov x22, %x[nb]\n"
                         "2:" // Block loop
                         "ldr q31, [%x[b_ptr], #0x0]\n"
                         "ldr q30, [%x[b_ptr], #0x10]\n"
                         "mov x21, x23\n"
                         "movi v29.4s, #0x0\n"
                         "ldr q28, [%x[b_ptr], #0x20]\n"
                         "ldr q27, [%x[b_ptr], #0x30]\n"
                         "movi v26.4s, #0x0\n"
                         "sub x20, x23, #0x2\n"
                         "ld1r { v25.8h }, [x20]\n"
                         "ldr q24, [%x[b_ptr], #-0x8]\n"
                         "sub x22, x22, #0x1\n"
                         "add x23, x23, #0x22\n"
                         "ld1r { v23.2d }, [x21], #0x8\n"
                         "sshl v22.16b, v31.16b, v2.16b\n"
                         "sshl v16.16b, v30.16b, v2.16b\n"
                         "add %x[b_ptr], %x[b_ptr], #0x48\n"
                         "ld1r { v21.2d }, [x21], #0x8\n"
                         "sshl v20.16b, v28.16b, v2.16b\n"
                         "sshl v19.16b, v27.16b, v2.16b\n"
                         "ld1r { v18.2d }, [x21], #0x8\n"
                         "ld1r { v17.2d }, [x21], #0x8\n"
                         "and v31.16b, v31.16b, v1.16b\n"
                         "and v30.16b, v30.16b, v1.16b\n"
                         ".inst 0x4e9796dd  // sdot v29.4s, v22.16b, v23.16b\n"
                         ".inst 0x4e97961a  // sdot v26.4s, v16.16b, v23.16b\n"
                         "and v28.16b, v28.16b, v1.16b\n"
                         "and v27.16b, v27.16b, v1.16b\n"
                         "fcvtl v25.4s, v25.4h\n"
                         "fcvtl v16.4s, v24.4h\n"
                         ".inst 0x4e95969d  // sdot v29.4s, v20.16b, v21.16b\n"
                         ".inst 0x4e95967a  // sdot v26.4s, v19.16b, v21.16b\n"
                         "fmul v16.4s, v16.4s, v25.4s\n"
                         ".inst 0x4e9297fd  // sdot v29.4s, v31.16b, v18.16b\n"
                         ".inst 0x4e9297da  // sdot v26.4s, v30.16b, v18.16b\n"
                         ".inst 0x4e91979d  // sdot v29.4s, v28.16b, v17.16b\n"
                         ".inst 0x4e91977a  // sdot v26.4s, v27.16b, v17.16b\n"
                         "addp v29.4s, v29.4s, v26.4s\n"
                         "scvtf v29.4s, v29.4s, #0x4\n"
                         "fmla v0.4s, v29.4s, v16.4s\n"
                         "cbnz x22, 2b\n"
                         "sub %x[nc], %x[nc], #0x4\n"
                         "str q0, [%x[res_ptr], #0x0]\n"
                         "add %x[res_ptr], %x[res_ptr], #0x10\n"
                         "cbnz %x[nc], 1b\n"
                         : [b_ptr] "+&r"(b_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
                         : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
                         : "memory", "v0", "v1", "v2", "v16", "v17", "v18", "v19", "v20", "v21",
                           "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                           "x20", "x21", "x22", "x23");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    assert((mllm_cpu_has_sve() || mllm_cpu_has_matmul_int8())
           && "__ARM_FEATURE_SVE and __ARM_FEATURE_MATMUL_INT8 not defined, use the "
              "Q4_0_4_4 quantization format for optimal "
              "performance");
#else
    float sumf[4];
    int sumi;

    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     << 4);
                        const int v1 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i])
                                 + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]))
                                >> 4;
                    }
                    sumf[j] +=
                        sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j]) * MLLM_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
#endif
}

void _gemv_q4_0_4x8_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc,
                              const void *__restrict bias) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

// #if defined(__ARM_FEATURE_SVE)
//     if (svcntw() == 8) {
//         assert(!(mllm_cpu_has_sve() && (svcntw() == 8))
//                && "__ARM_FEATURE_SVE defined, use the Q4_0_8_8 quantization format "
//                   "for optimal performance");
//     }
// #endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    const void *bias_ptr = bias;
    float *res_ptr = s;

    __asm__ __volatile__(
        "movi v2.16b, #0x4\n"
        "movi v1.16b, #0xf0\n"
        "add %x[b_ptr], %x[b_ptr], #0x8\n"
        "1:" // Column loop
        "add x23, %x[a_ptr], #0x2\n"
        "movi v0.16b, #0x0\n"
        "mov x22, %x[nb]\n"
        "2:" // Block loop
        "ldr q31, [%x[b_ptr], #0x0]\n"
        "ldr q30, [%x[b_ptr], #0x10]\n"
        "mov x21, x23\n"
        "movi v29.4s, #0x0\n"
        "ldr q28, [%x[b_ptr], #0x20]\n"
        "ldr q27, [%x[b_ptr], #0x30]\n"
        "movi v26.4s, #0x0\n"
        "sub x20, x23, #0x2\n"
        "ld1r { v25.8h }, [x20]\n"
        "ldr q24, [%x[b_ptr], #-0x8]\n"
        "sub x22, x22, #0x1\n"
        "add x23, x23, #0x22\n"
        "ld1r { v23.2d }, [x21], #0x8\n"
        "sshl v22.16b, v31.16b, v2.16b\n"
        "sshl v16.16b, v30.16b, v2.16b\n"
        "add %x[b_ptr], %x[b_ptr], #0x48\n"
        "ld1r { v21.2d }, [x21], #0x8\n"
        "sshl v20.16b, v28.16b, v2.16b\n"
        "sshl v19.16b, v27.16b, v2.16b\n"
        "ld1r { v18.2d }, [x21], #0x8\n"
        "ld1r { v17.2d }, [x21], #0x8\n"
        "and v31.16b, v31.16b, v1.16b\n"
        "and v30.16b, v30.16b, v1.16b\n"
        ".inst 0x4e9796dd  // sdot v29.4s, v22.16b, v23.16b\n"
        ".inst 0x4e97961a  // sdot v26.4s, v16.16b, v23.16b\n"
        "and v28.16b, v28.16b, v1.16b\n"
        "and v27.16b, v27.16b, v1.16b\n"
        "fcvtl v25.4s, v25.4h\n"
        "fcvtl v16.4s, v24.4h\n"
        ".inst 0x4e95969d  // sdot v29.4s, v20.16b, v21.16b\n"
        ".inst 0x4e95967a  // sdot v26.4s, v19.16b, v21.16b\n"
        "fmul v16.4s, v16.4s, v25.4s\n"
        ".inst 0x4e9297fd  // sdot v29.4s, v31.16b, v18.16b\n"
        ".inst 0x4e9297da  // sdot v26.4s, v30.16b, v18.16b\n"
        ".inst 0x4e91979d  // sdot v29.4s, v28.16b, v17.16b\n"
        ".inst 0x4e91977a  // sdot v26.4s, v27.16b, v17.16b\n"
        "addp v29.4s, v29.4s, v26.4s\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "fmla v0.4s, v29.4s, v16.4s\n"
        "cbnz x22, 2b\n"
        "sub %x[nc], %x[nc], #0x4\n"
        // -- bias start
        "ldr q28, [%x[bias_ptr], #0x0]\n"         // load data from bias_ptr
        "fadd v0.4s, v0.4s, v28.4s\n"             // add q0 = q0 + q28
        "add %x[bias_ptr], %x[bias_ptr], #0x10\n" // move bias_ptr to next side.
        // -- bias end
        "str q0, [%x[res_ptr], #0x0]\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "cbnz %x[nc], 1b\n"
        : [b_ptr] "+&r"(b_ptr), [bias_ptr] "+&r"(bias_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
        : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
        : "memory", "v0", "v1", "v2", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
          "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    assert((mllm_cpu_has_sve() || mllm_cpu_has_matmul_int8())
           && "__ARM_FEATURE_SVE and __ARM_FEATURE_MATMUL_INT8 not defined, use the "
              "Q4_0_4_4 quantization format for optimal "
              "performance");
#else
    float sumf[4];
    int sumi;

    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    const float *bias_ptr = (const float *)bias;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = bias_ptr[x * ncols_interleaved + j];
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     << 4);
                        const int v1 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i])
                                 + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]))
                                >> 4;
                    }
                    sumf[j] +=
                        sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j]) * MLLM_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
#endif
}

void gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                        const void *__restrict vy, int nr, int nc,
                        const void *__restrict bias) {
    if (bias != nullptr) {
        _gemv_q4_0_8x8_q8_0_bias(n, s, bs, vx, vy, nr, nc, bias);
        return;
    }

    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

#if defined(__ARM_FEATURE_SVE)
    if (svcntw() == 8) {
        // if (true) {
        const void *b_ptr = vx;
        const void *a_ptr = vy;
        float *res_ptr = s;

        __asm__ __volatile__("ptrue p0.b\n"
                             "add %x[b_ptr], %x[b_ptr], #0x10\n"
                             "1:" // Column loop
                             "add x22, %x[a_ptr], #0x2\n"
                             "mov z31.b, #0x0\n"
                             "mov x21, %x[nb]\n"
                             "2:" // Block loop
                             "ld1b { z30.b }, p0/Z, [%x[b_ptr]]\n"
                             "ld1b { z29.b }, p0/Z, [%x[b_ptr], #1, MUL VL]\n"
                             "mov z28.s, #0x0\n"
                             "mov z27.s, #0x0\n"
                             "ld1rd { z26.d }, p0/Z, [x22]\n"
                             "ld1b { z25.b }, p0/Z, [%x[b_ptr], #2, MUL VL]\n"
                             "sub x20, x22, #0x2\n"
                             "sub x21, x21, #0x1\n"
                             "ld1b { z24.b }, p0/Z, [%x[b_ptr], #3, MUL VL]\n"
                             "ld1rd { z23.d }, p0/Z, [x22, #8]\n"
                             "lsl z22.b, z30.b, #0x4\n"
                             "lsl z16.b, z29.b, #0x4\n"
                             "and z30.b, z30.b, #0xf0\n"
                             "and z29.b, z29.b, #0xf0\n"
                             "ld1rd { z21.d }, p0/Z, [x22, #16]\n"
                             "ld1rd { z20.d }, p0/Z, [x22, #24]\n"
                             "lsl z19.b, z25.b, #0x4\n"
                             "and z25.b, z25.b, #0xf0\n"
                             "ld1rh { z17.h }, p0/Z, [x20]\n"
                             "ld1h { z18.s }, p0/Z, [%x[b_ptr], #-1, MUL VL]\n"
                             "sdot z28.s, z22.b, z26.b\n"
                             "sdot z27.s, z16.b, z26.b\n"
                             "lsl z16.b, z24.b, #0x4\n"
                             "add x22, x22, #0x22\n"
                             "and z24.b, z24.b, #0xf0\n"
                             "add %x[b_ptr], %x[b_ptr], #0x90\n"
                             "fcvt z17.s, p0/m, z17.h\n"
                             "fcvt z18.s, p0/m, z18.h\n"
                             "sdot z28.s, z19.b, z23.b\n"
                             "sdot z27.s, z16.b, z23.b\n"
                             "fmul z18.s, z18.s, z17.s\n"
                             "sdot z28.s, z30.b, z21.b\n"
                             "sdot z27.s, z29.b, z21.b\n"
                             "sdot z28.s, z25.b, z20.b\n"
                             "sdot z27.s, z24.b, z20.b\n"
                             "uzp1 z17.s, z28.s, z27.s\n"
                             "uzp2 z16.s, z28.s, z27.s\n"
                             "add z17.s, z17.s, z16.s\n"
                             "asr z17.s, z17.s, #0x4\n"
                             "scvtf z17.s, p0/m, z17.s\n"
                             "fmla z31.s, p0/M, z17.s, z18.s\n"
                             "cbnz x21, 2b\n"
                             "sub %x[nc], %x[nc], #0x8\n"
                             "st1w { z31.s }, p0, [%x[res_ptr]]\n"
                             "add %x[res_ptr], %x[res_ptr], #0x20\n"
                             "cbnz %x[nc], 1b\n"
                             : [b_ptr] "+&r"(b_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
                             : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
                             : "memory", "p0", "x20", "x21", "x22", "z16", "z17", "z18", "z19",
                               "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
                               "z30", "z31");
        return;
    }
    // else if (mllm_cpu_has_neon() && mllm_cpu_has_matmul_int8()) {
    //     assert((mllm_cpu_has_sve() && (svcntw() == 8))
    //            && "__ARM_FEATURE_SVE for vector size of 256-bits not defined, use the "
    //               "Q4_0_4_8 quantization format for optimal "
    //               "performance");
    // } else if (mllm_cpu_has_neon()) {
    //     assert(((mllm_cpu_has_sve() && (svcntw() == 8)) || mllm_cpu_has_matmul_int8())
    //            && "__ARM_FEATURE_SVE for vector size of 256-bits and "
    //               "__ARM_FEATURE_MATMUL_INT8 not defined, use the Q4_0_4_4 "
    //               "quantization format for optimal performance");
    // }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    assert(mllm_cpu_has_sve()
           && "__ARM_FEATURE_SVE not defined, use the Q4_0_4_8 quantization format "
              "for optimal performance");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    assert((mllm_cpu_has_sve() || mllm_cpu_has_matmul_int8())
           && "__ARM_FEATURE_SVE and __ARM_FEATURE_MATMUL_INT8 not defined, use the "
              "Q4_0_4_4 quantization format for optimal "
              "performance");
#else
    float sumf[8];
    int sumi;

    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     << 4);
                        const int v1 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i])
                                 + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]))
                                >> 4;
                    }
                    sumf[j] +=
                        sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j]) * MLLM_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
#endif
}

void _gemv_q4_0_8x8_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc,
                              const void *__restrict bias) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

#if defined(__ARM_FEATURE_SVE)
    // if (svcntw() == 8) {
    if (true) {
        const void *b_ptr = vx;
        const void *a_ptr = vy;
        const void *bias_ptr = bias;
        float *res_ptr = s;

        __asm__ __volatile__("ptrue p0.b\n"
                             "add %x[b_ptr], %x[b_ptr], #0x10\n"
                             "1:" // Column loop
                             "add x22, %x[a_ptr], #0x2\n"
                             "mov z31.b, #0x0\n"
                             "mov x21, %x[nb]\n"
                             "2:" // Block loop
                             "ld1b { z30.b }, p0/Z, [%x[b_ptr]]\n"
                             "ld1b { z29.b }, p0/Z, [%x[b_ptr], #1, MUL VL]\n"
                             "mov z28.s, #0x0\n"
                             "mov z27.s, #0x0\n"
                             "ld1rd { z26.d }, p0/Z, [x22]\n"
                             "ld1b { z25.b }, p0/Z, [%x[b_ptr], #2, MUL VL]\n"
                             "sub x20, x22, #0x2\n"
                             "sub x21, x21, #0x1\n"
                             "ld1b { z24.b }, p0/Z, [%x[b_ptr], #3, MUL VL]\n"
                             "ld1rd { z23.d }, p0/Z, [x22, #8]\n"
                             "lsl z22.b, z30.b, #0x4\n"
                             "lsl z16.b, z29.b, #0x4\n"
                             "and z30.b, z30.b, #0xf0\n"
                             "and z29.b, z29.b, #0xf0\n"
                             "ld1rd { z21.d }, p0/Z, [x22, #16]\n"
                             "ld1rd { z20.d }, p0/Z, [x22, #24]\n"
                             "lsl z19.b, z25.b, #0x4\n"
                             "and z25.b, z25.b, #0xf0\n"
                             "ld1rh { z17.h }, p0/Z, [x20]\n"
                             "ld1h { z18.s }, p0/Z, [%x[b_ptr], #-1, MUL VL]\n"
                             "sdot z28.s, z22.b, z26.b\n"
                             "sdot z27.s, z16.b, z26.b\n"
                             "lsl z16.b, z24.b, #0x4\n"
                             "add x22, x22, #0x22\n"
                             "and z24.b, z24.b, #0xf0\n"
                             "add %x[b_ptr], %x[b_ptr], #0x90\n"
                             "fcvt z17.s, p0/m, z17.h\n"
                             "fcvt z18.s, p0/m, z18.h\n"
                             "sdot z28.s, z19.b, z23.b\n"
                             "sdot z27.s, z16.b, z23.b\n"
                             "fmul z18.s, z18.s, z17.s\n"
                             "sdot z28.s, z30.b, z21.b\n"
                             "sdot z27.s, z29.b, z21.b\n"
                             "sdot z28.s, z25.b, z20.b\n"
                             "sdot z27.s, z24.b, z20.b\n"
                             "uzp1 z17.s, z28.s, z27.s\n"
                             "uzp2 z16.s, z28.s, z27.s\n"
                             "add z17.s, z17.s, z16.s\n"
                             "asr z17.s, z17.s, #0x4\n"
                             "scvtf z17.s, p0/m, z17.s\n"
                             "fmla z31.s, p0/M, z17.s, z18.s\n"
                             "cbnz x21, 2b\n"
                             "sub %x[nc], %x[nc], #0x8\n"
                             // --bias code start
                             "ld1b { z17.b }, p0/Z, [%x[bias_ptr]]\n"
                             "fadd z31.s, p0/M, z31.s, z17.s\n"
                             "add %x[bias_ptr], %x[bias_ptr], #0x20\n"
                             // --bias code end
                             "st1w { z31.s }, p0, [%x[res_ptr]]\n"
                             "add %x[res_ptr], %x[res_ptr], #0x20\n"
                             "cbnz %x[nc], 1b\n"
                             : [b_ptr] "+&r"(b_ptr), [bias_ptr] "+&r"(bias_ptr),
                               [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
                             : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
                             : "memory", "p0", "x20", "x21", "x22", "z16", "z17", "z18", "z19",
                               "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
                               "z30", "z31");
        return;
    }

    // else if (mllm_cpu_has_neon() && mllm_cpu_has_matmul_int8()) {
    //     assert((mllm_cpu_has_sve() && (svcntw() == 8))
    //            && "__ARM_FEATURE_SVE for vector size of 256-bits not defined, use the "
    //               "Q4_0_4_8 quantization format for optimal "
    //               "performance");
    // } else if (mllm_cpu_has_neon()) {
    //     assert(((mllm_cpu_has_sve() && (svcntw() == 8)) || mllm_cpu_has_matmul_int8())
    //            && "__ARM_FEATURE_SVE for vector size of 256-bits and "
    //               "__ARM_FEATURE_MATMUL_INT8 not defined, use the Q4_0_4_4 "
    //               "quantization format for optimal performance");
    // }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    assert(mllm_cpu_has_sve()
           && "__ARM_FEATURE_SVE not defined, use the Q4_0_4_8 quantization format "
              "for optimal performance");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    assert((mllm_cpu_has_sve() || mllm_cpu_has_matmul_int8())
           && "__ARM_FEATURE_SVE and __ARM_FEATURE_MATMUL_INT8 not defined, use the "
              "Q4_0_4_4 quantization format for optimal "
              "performance");
#else
    float sumf[8];
    int sumi;

    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    const float *bias_ptr = (const float *)bias;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = bias_ptr[x * ncols_interleaved + j];
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     << 4);
                        const int v1 =
                            (int8_t)(b_ptr[l]
                                         .qs[k * ncols_interleaved * blocklen + j * blocklen + i]
                                     & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i])
                                 + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]))
                                >> 4;
                    }
                    sumf[j] +=
                        sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j]) * MLLM_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
#endif
}

// lhs: q8_0, rhs: q4_0x4
void gemm_q4_0_4x4_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                        const void *__restrict vy, int nr, int nc,
                        const void *__restrict bias) {
    if (bias != nullptr) {
        _gemm_q4_0_4x4_q8_0_bias(n, s, bs, vx, vy, nr, nc, bias);
#if defined(__ARM_NEON)
        std::cout << "_gemm_q4_0_4x4_q8_0_bias not implemented";
        abort();
#endif
        return;
    }

    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

// #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
//     if (svcntw() == 8) {
//         assert(!(mllm_cpu_has_sve() && (svcntw() == 8))
//                && "__ARM_FEATURE_SVE defined, use the Q4_0_8_8 quantization format "
//                   "for optimal performance");
//     }
// #endif
// #if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
//     assert(!(mllm_cpu_has_neon() && mllm_cpu_has_matmul_int8())
//            && "__ARM_NEON and __ARM_FEATURE_MATMUL_INT8 defined, use the Q4_0_4_8 "
//               "quantization format for optimal performance");
// #elif defined(__ARM_NEON) && defined(__aarch64__)
#if defined(__ARM_NEON) && defined(__aarch64__)
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    float *res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[nb], x9\n"
        "blt 4f\n"
        "1:" // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[nc]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:" // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "mov x24, %x[nb]\n"
        "add x23, x25, x9\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v23.16b, #0x0\n"
        "movi v16.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v0.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v21.16b, #0x0\n"
        "movi v8.16b, #0x0\n"
        "movi v1.16b, #0x0\n"
        "3:" // Block loop
        "ldr q3, [x28, #0x0]\n"
        "ldr q31, [x25, #0x0]\n"
        "movi v28.16b, #0x4\n"
        "movi v10.4s, #0x0\n"
        "ldr q22, [x28, #0x10]\n"
        "ldr q6, [x25, #0x10]\n"
        "movi v29.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "ldr q27, [x28, #0x20]\n"
        "ldr q30, [x28, #0x30]\n"
        "movi v20.4s, #0x0\n"
        "movi v24.16b, #0xf0\n"
        "ldr d2, [x25, #-0x8]\n"
        "ldr d26, [x23, #-0x8]\n"
        "sshl v12.16b, v3.16b, v28.16b\n"
        "sub x20, x28, #0x8\n"
        "ldr d17, [x20, #0x0]\n"
        "and v3.16b, v3.16b, v24.16b\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]\n"
        ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]\n"
        ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]\n"
        ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]\n"
        "sshl v31.16b, v22.16b, v28.16b\n"
        "and v22.16b, v22.16b, v24.16b\n"
        "fcvtl v17.4s, v17.4h\n"
        "fcvtl v2.4s, v2.4h\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]\n"
        ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]\n"
        ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]\n"
        ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]\n"
        "sshl v6.16b, v27.16b, v28.16b\n"
        "sshl v28.16b, v30.16b, v28.16b\n"
        "and v27.16b, v27.16b, v24.16b\n"
        "and v30.16b, v30.16b, v24.16b\n"
        "ldr q24, [x25, #0x20]\n"
        ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x30]\n"
        ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]\n"
        ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]\n"
        ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x40]\n"
        ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x50]\n"
        ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]\n"
        ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]\n"
        ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]\n"
        ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x60]\n"
        ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]\n"
        ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]\n"
        ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]\n"
        ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]\n"
        "fmul v24.4s, v17.4s, v2.s[0]\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v15.4s, v10.4s, v24.4s\n"
        "ldr q24, [x23, #0x0]\n"
        "fmul v10.4s, v17.4s, v2.s[1]\n"
        "fmla v19.4s, v29.4s, v10.4s\n"
        "ldr q10, [x23, #0x10]\n"
        "fmul v29.4s, v17.4s, v2.s[2]\n"
        "fmul v2.4s, v17.4s, v2.s[3]\n"
        "fmla v18.4s, v9.4s, v29.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]\n"
        "fmla v14.4s, v20.4s, v2.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x20]\n"
        ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]\n"
        ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]\n"
        ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]\n"
        ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x30]\n"
        ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x40]\n"
        ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]\n"
        ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]\n"
        ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]\n"
        ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x50]\n"
        ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x60]\n"
        ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]\n"
        ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]\n"
        ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]\n"
        ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x0]\n"
        ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]\n"
        ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]\n"
        ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]\n"
        ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]\n"
        "fmul v10.4s, v17.4s, v26.s[0]\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v11.4s, v9.4s, v10.4s\n"
        "ldr q9, [x22, #0x10]\n"
        "fmul v10.4s, v17.4s, v26.s[1]\n"
        "fmla v13.4s, v29.4s, v10.4s\n"
        "ldr d29, [x22, #-0x8]\n"
        "fmul v10.4s, v17.4s, v26.s[2]\n"
        "fmul v26.4s, v17.4s, v26.s[3]\n"
        "fcvtl v29.4s, v29.4h\n"
        "fmla v23.4s, v20.4s, v10.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v16.4s, v2.4s, v26.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x20]\n"
        ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x30]\n"
        ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x40]\n"
        ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x50]\n"
        ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x60]\n"
        ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x21, #0x0]\n"
        ".inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]\n"
        ".inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]\n"
        "fmul v9.4s, v17.4s, v29.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v25.4s, v20.4s, v9.4s\n"
        "ldr q9, [x21, #0x10]\n"
        "fmul v20.4s, v17.4s, v29.s[1]\n"
        "fmla v7.4s, v10.4s, v20.4s\n"
        "ldr d20, [x21, #-0x8]\n"
        "fmul v10.4s, v17.4s, v29.s[2]\n"
        "fmul v29.4s, v17.4s, v29.s[3]\n"
        "fcvtl v20.4s, v20.4h\n"
        "fmla v0.4s, v26.4s, v10.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v4.4s, v2.4s, v29.4s\n"
        "movi v2.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]\n"
        "ldr q12, [x21, #0x20]\n"
        "fmul v24.4s, v17.4s, v20.s[0]\n"
        ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x30]\n"
        "fmul v31.4s, v17.4s, v20.s[1]\n"
        ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]\n"
        ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]\n"
        ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]\n"
        ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x40]\n"
        "fmul v6.4s, v17.4s, v20.s[2]\n"
        "fmul v20.4s, v17.4s, v20.s[3]\n"
        ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x50]\n"
        ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]\n"
        ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]\n"
        ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]\n"
        ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x60]\n"
        ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]\n"
        "ldr q17, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]\n"
        ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]\n"
        ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]\n"
        ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]\n"
        ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]\n"
        ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]\n"
        ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]\n"
        ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "fmla v5.4s, v26.4s, v24.4s\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "fmla v21.4s, v10.4s, v31.4s\n"
        "fmla v8.4s, v2.4s, v6.4s\n"
        "fmla v1.4s, v29.4s, v20.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q16, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q0, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q21, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q8, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q1, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:" // Row loop skip
        "cbz x10, 9f\n"
        "5:" // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[nc]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:" // Row tail: Column loop
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[nb]\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "7:" // Row tail: Block loop
        "ldr q7, [x24, #0x0]\n"
        "ldr q5, [x25, #0x0]\n"
        "movi v9.16b, #0x4\n"
        "movi v4.4s, #0x0\n"
        "ldr q3, [x24, #0x10]\n"
        "ldr q2, [x25, #0x10]\n"
        "movi v1.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q13, [x24, #0x20]\n"
        "ldr q31, [x25, #0x20]\n"
        "movi v30.4s, #0x0\n"
        "movi v29.16b, #0xf0\n"
        "ldr q28, [x24, #0x30]\n"
        "ldr q27, [x25, #0x30]\n"
        "sshl v20.16b, v7.16b, v9.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr q26, [x25, #0x40]\n"
        "ldr q25, [x25, #0x50]\n"
        "sshl v17.16b, v3.16b, v9.16b\n"
        "and v7.16b, v7.16b, v29.16b\n"
        "ldr q24, [x25, #0x60]\n"
        "ldr q16, [x25, #0x70]\n"
        "sshl v22.16b, v13.16b, v9.16b\n"
        "and v3.16b, v3.16b, v29.16b\n"
        "ldr d21, [x20, #0x0]\n"
        "ldr d12, [x25, #-0x8]\n"
        ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]\n"
        ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]\n"
        ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]\n"
        ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]\n"
        "sshl v9.16b, v28.16b, v9.16b\n"
        "subs x21, x21, #0x1\n"
        "and v13.16b, v13.16b, v29.16b\n"
        "and v28.16b, v28.16b, v29.16b\n"
        "add x25, x25, #0x88\n"
        "add x24, x24, #0x48\n"
        "fcvtl v21.4s, v21.4h\n"
        "fcvtl v12.4s, v12.4h\n"
        ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]\n"
        ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]\n"
        ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]\n"
        ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]\n"
        "fmul v11.4s, v21.4s, v12.s[0]\n"
        "fmul v23.4s, v21.4s, v12.s[1]\n"
        "fmul v17.4s, v21.4s, v12.s[2]\n"
        ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]\n"
        "fmul v6.4s, v21.4s, v12.s[3]\n"
        ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]\n"
        ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]\n"
        ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]\n"
        ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]\n"
        ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]\n"
        ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]\n"
        ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]\n"
        ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]\n"
        ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]\n"
        ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
        ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]\n"
        ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]\n"
        ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]\n"
        ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]\n"
        ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]\n"
        ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]\n"
        ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]\n"
        ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]\n"
        ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]\n"
        ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]\n"
        ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]\n"
        ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]\n"
        ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "scvtf v1.4s, v1.4s, #0x4\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "fmla v15.4s, v4.4s, v11.4s\n"
        "scvtf v30.4s, v30.4s, #0x4\n"
        "fmla v19.4s, v1.4s, v23.4s\n"
        "fmla v18.4s, v0.4s, v17.4s\n"
        "fmla v14.4s, v30.4s, v6.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q14, [x20, #0x0]\n"
        "8:" // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:" // Row tail: Row loop skip
        : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
        : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb), [res_stride] "r"(res_stride), [nc] "r"(nc)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
          "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
          "x24", "x25", "x26", "x27", "x28");
#else
    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        << 4);
                                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i])
                                         + (v1
                                            * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i
                                                          + qk / 2 * 4]))
                                        >> 4;
                            }
                            sumf[m][j] += sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j])
                                          * MLLM_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
#endif
}

void _gemm_q4_0_4x4_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc,
                              const void *__restrict bias) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

// #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
//     if (svcntw() == 8) {
//         assert(!(mllm_cpu_has_sve() && (svcntw() == 8))
//                && "__ARM_FEATURE_SVE defined, use the Q4_0_8_8 quantization format "
//                   "for optimal performance");
//     }
// #endif
// #if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
//     assert(!(mllm_cpu_has_neon() && mllm_cpu_has_matmul_int8())
//            && "__ARM_NEON and __ARM_FEATURE_MATMUL_INT8 defined, use the Q4_0_4_8 "
//               "quantization format for optimal performance");
// #elif defined(__ARM_NEON) && defined(__aarch64__)
#if defined(__ARM_NEON) && defined(__aarch64__)
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    const void *bias_ptr = bias;
    float *res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[nb], x9\n"
        "blt 4f\n"
        "1:" // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[nc]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:" // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "mov x24, %x[nb]\n"
        "add x23, x25, x9\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v23.16b, #0x0\n"
        "movi v16.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v0.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v21.16b, #0x0\n"
        "movi v8.16b, #0x0\n"
        "movi v1.16b, #0x0\n"
        "3:" // Block loop
        "ldr q3, [x28, #0x0]\n"
        "ldr q31, [x25, #0x0]\n"
        "movi v28.16b, #0x4\n"
        "movi v10.4s, #0x0\n"
        "ldr q22, [x28, #0x10]\n"
        "ldr q6, [x25, #0x10]\n"
        "movi v29.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "ldr q27, [x28, #0x20]\n"
        "ldr q30, [x28, #0x30]\n"
        "movi v20.4s, #0x0\n"
        "movi v24.16b, #0xf0\n"
        "ldr d2, [x25, #-0x8]\n"
        "ldr d26, [x23, #-0x8]\n"
        "sshl v12.16b, v3.16b, v28.16b\n"
        "sub x20, x28, #0x8\n"
        "ldr d17, [x20, #0x0]\n"
        "and v3.16b, v3.16b, v24.16b\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]\n"
        ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]\n"
        ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]\n"
        ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]\n"
        "sshl v31.16b, v22.16b, v28.16b\n"
        "and v22.16b, v22.16b, v24.16b\n"
        "fcvtl v17.4s, v17.4h\n"
        "fcvtl v2.4s, v2.4h\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]\n"
        ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]\n"
        ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]\n"
        ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]\n"
        "sshl v6.16b, v27.16b, v28.16b\n"
        "sshl v28.16b, v30.16b, v28.16b\n"
        "and v27.16b, v27.16b, v24.16b\n"
        "and v30.16b, v30.16b, v24.16b\n"
        "ldr q24, [x25, #0x20]\n"
        ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x30]\n"
        ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]\n"
        ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]\n"
        ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x40]\n"
        ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x50]\n"
        ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]\n"
        ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]\n"
        ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]\n"
        ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x60]\n"
        ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]\n"
        ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]\n"
        ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]\n"
        ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]\n"
        "fmul v24.4s, v17.4s, v2.s[0]\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v15.4s, v10.4s, v24.4s\n"
        "ldr q24, [x23, #0x0]\n"
        "fmul v10.4s, v17.4s, v2.s[1]\n"
        "fmla v19.4s, v29.4s, v10.4s\n"
        "ldr q10, [x23, #0x10]\n"
        "fmul v29.4s, v17.4s, v2.s[2]\n"
        "fmul v2.4s, v17.4s, v2.s[3]\n"
        "fmla v18.4s, v9.4s, v29.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]\n"
        "fmla v14.4s, v20.4s, v2.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x20]\n"
        ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]\n"
        ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]\n"
        ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]\n"
        ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x30]\n"
        ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x40]\n"
        ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]\n"
        ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]\n"
        ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]\n"
        ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x50]\n"
        ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x60]\n"
        ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]\n"
        ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]\n"
        ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]\n"
        ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x0]\n"
        ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]\n"
        ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]\n"
        ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]\n"
        ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]\n"
        "fmul v10.4s, v17.4s, v26.s[0]\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v11.4s, v9.4s, v10.4s\n"
        "ldr q9, [x22, #0x10]\n"
        "fmul v10.4s, v17.4s, v26.s[1]\n"
        "fmla v13.4s, v29.4s, v10.4s\n"
        "ldr d29, [x22, #-0x8]\n"
        "fmul v10.4s, v17.4s, v26.s[2]\n"
        "fmul v26.4s, v17.4s, v26.s[3]\n"
        "fcvtl v29.4s, v29.4h\n"
        "fmla v23.4s, v20.4s, v10.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v16.4s, v2.4s, v26.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x20]\n"
        ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x30]\n"
        ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x40]\n"
        ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x50]\n"
        ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x60]\n"
        ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x21, #0x0]\n"
        ".inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]\n"
        ".inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]\n"
        "fmul v9.4s, v17.4s, v29.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v25.4s, v20.4s, v9.4s\n"
        "ldr q9, [x21, #0x10]\n"
        "fmul v20.4s, v17.4s, v29.s[1]\n"
        "fmla v7.4s, v10.4s, v20.4s\n"
        "ldr d20, [x21, #-0x8]\n"
        "fmul v10.4s, v17.4s, v29.s[2]\n"
        "fmul v29.4s, v17.4s, v29.s[3]\n"
        "fcvtl v20.4s, v20.4h\n"
        "fmla v0.4s, v26.4s, v10.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v4.4s, v2.4s, v29.4s\n"
        "movi v2.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]\n"
        "ldr q12, [x21, #0x20]\n"
        "fmul v24.4s, v17.4s, v20.s[0]\n"
        ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x30]\n"
        "fmul v31.4s, v17.4s, v20.s[1]\n"
        ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]\n"
        ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]\n"
        ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]\n"
        ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x40]\n"
        "fmul v6.4s, v17.4s, v20.s[2]\n"
        "fmul v20.4s, v17.4s, v20.s[3]\n"
        ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x50]\n"
        ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]\n"
        ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]\n"
        ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]\n"
        ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x60]\n"
        ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]\n"
        "ldr q17, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]\n"
        ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]\n"
        ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]\n"
        ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]\n"
        ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]\n"
        ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]\n"
        ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]\n"
        ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "fmla v5.4s, v26.4s, v24.4s\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "fmla v21.4s, v10.4s, v31.4s\n"
        "fmla v8.4s, v2.4s, v6.4s\n"
        "fmla v1.4s, v29.4s, v20.4s\n"
        "bgt 3b\n"
        // --bias for tiled part, begin
        // "mov x20, %x[bias_ptr]\n" // 将 bias_ptr 移动到 x21 寄存器

        "ldr q28, [%x[bias_ptr], #0x0]\n"         // 从 bias_ptr 读取数据到 q28
        "fadd v15.4s, v15.4s, v28.4s\n"           // 将 q28 加到 v15 上
        "add %x[bias_ptr], %x[bias_ptr], #0x10\n" // 更新 bias_ptr，偏移到下一个位置

        "ldr q28, [%x[bias_ptr], #0x0]\n"         // 从 bias_ptr 读取数据到 q28
        "fadd v19.4s, v19.4s, v28.4s\n"           // 将 q28 加到 v19 上
        "add %x[bias_ptr], %x[bias_ptr], #0x10\n" // 更新 bias_ptr，偏移到下一个位置

        "ldr q28, [%x[bias_ptr], #0x0]\n"         // 从 bias_ptr 读取数据到 q28
        "fadd v18.4s, v18.4s, v28.4s\n"           // 将 q28 加到 v18 上
        "add %x[bias_ptr], %x[bias_ptr], #0x10\n" // 更新 bias_ptr，偏移到下一个位置

        "ldr q28, [%x[bias_ptr], #0x0]\n"         // 从 bias_ptr 读取数据到 q28
        "fadd v14.4s, v14.4s, v28.4s\n"           // 将 q28 加到 v14 上
        "add %x[bias_ptr], %x[bias_ptr], #0x10\n" // 更新 bias_ptr，偏移到下一个位置

        // TODO find a correct register to get data from bias_ptr.
        // --bias for tiled part, end
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q16, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q0, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q21, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q8, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q1, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:" // Row loop skip
        "cbz x10, 9f\n"
        "5:" // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[nc]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:" // Row tail: Column loop
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[nb]\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "7:" // Row tail: Block loop
        "ldr q7, [x24, #0x0]\n"
        "ldr q5, [x25, #0x0]\n"
        "movi v9.16b, #0x4\n"
        "movi v4.4s, #0x0\n"
        "ldr q3, [x24, #0x10]\n"
        "ldr q2, [x25, #0x10]\n"
        "movi v1.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q13, [x24, #0x20]\n"
        "ldr q31, [x25, #0x20]\n"
        "movi v30.4s, #0x0\n"
        "movi v29.16b, #0xf0\n"
        "ldr q28, [x24, #0x30]\n"
        "ldr q27, [x25, #0x30]\n"
        "sshl v20.16b, v7.16b, v9.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr q26, [x25, #0x40]\n"
        "ldr q25, [x25, #0x50]\n"
        "sshl v17.16b, v3.16b, v9.16b\n"
        "and v7.16b, v7.16b, v29.16b\n"
        "ldr q24, [x25, #0x60]\n"
        "ldr q16, [x25, #0x70]\n"
        "sshl v22.16b, v13.16b, v9.16b\n"
        "and v3.16b, v3.16b, v29.16b\n"
        "ldr d21, [x20, #0x0]\n"
        "ldr d12, [x25, #-0x8]\n"
        ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]\n"
        ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]\n"
        ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]\n"
        ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]\n"
        "sshl v9.16b, v28.16b, v9.16b\n"
        "subs x21, x21, #0x1\n"
        "and v13.16b, v13.16b, v29.16b\n"
        "and v28.16b, v28.16b, v29.16b\n"
        "add x25, x25, #0x88\n"
        "add x24, x24, #0x48\n"
        "fcvtl v21.4s, v21.4h\n"
        "fcvtl v12.4s, v12.4h\n"
        ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]\n"
        ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]\n"
        ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]\n"
        ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]\n"
        "fmul v11.4s, v21.4s, v12.s[0]\n"
        "fmul v23.4s, v21.4s, v12.s[1]\n"
        "fmul v17.4s, v21.4s, v12.s[2]\n"
        ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]\n"
        "fmul v6.4s, v21.4s, v12.s[3]\n"
        ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]\n"
        ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]\n"
        ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]\n"
        ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]\n"
        ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]\n"
        ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]\n"
        ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]\n"
        ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]\n"
        ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]\n"
        ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
        ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]\n"
        ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]\n"
        ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]\n"
        ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]\n"
        ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]\n"
        ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]\n"
        ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]\n"
        ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]\n"
        ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]\n"
        ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]\n"
        ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]\n"
        ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]\n"
        ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "scvtf v1.4s, v1.4s, #0x4\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "fmla v15.4s, v4.4s, v11.4s\n"
        "scvtf v30.4s, v30.4s, #0x4\n"
        "fmla v19.4s, v1.4s, v23.4s\n"
        "fmla v18.4s, v0.4s, v17.4s\n"
        "fmla v14.4s, v30.4s, v6.4s\n"
        "bgt 7b\n"
        // --bias start

        // --bias end
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q14, [x20, #0x0]\n"
        "8:" // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:" // Row tail: Row loop skip
        : [a_ptr] "+&r"(a_ptr), [bias_ptr] "+&r"(bias_ptr), [res_ptr] "+&r"(res_ptr)
        : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb), [res_stride] "r"(res_stride), [nc] "r"(nc)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
          "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
          "x24", "x25", "x26", "x27", "x28");
#else
    float sumf[4][4];
    int sumi;

    const float *bias_ptr = (const float *)bias;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    sumf[m][j] = bias_ptr[x * ncols_interleaved + j];
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        << 4);
                                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i])
                                         + (v1
                                            * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i
                                                          + qk / 2 * 4]))
                                        >> 4;
                            }
                            sumf[m][j] += sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j])
                                          * MLLM_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
#endif
}

void gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                        const void *__restrict vy, int nr, int nc,
                        const void *__restrict bias) {
    if (bias != nullptr) {
#if defined(__ARM_NEON)
        std::cout << "_gemm_q4_0_4x8_q8_0_bias not implemented";
        abort();
#endif
        _gemm_q4_0_4x8_q8_0_bias(n, s, bs, vx, vy, nr, nc, bias);
        return;
    }

    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

// #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
//     if (svcntw() == 8) {
//         assert(!(mllm_cpu_has_sve() && (svcntw() == 8))
//                && "__ARM_FEATURE_SVE defined, use the Q4_0_8_8 quantization format "
//                   "for optimal performance");
//     }
// #endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    float *res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[nb], x9\n"
        "blt 4f\n"
        "1:" // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[nc]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:" // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "mov x24, %x[nb]\n"
        "add x23, x25, x9\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v22.16b, #0x0\n"
        "movi v23.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v6.16b, #0x0\n"
        "movi v30.16b, #0x0\n"
        "movi v24.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "3:" // Block loop
        "ldr q21, [x28, #0x0]\n"
        "ldr q16, [x28, #0x10]\n"
        "movi v1.16b, #0x4\n"
        "movi v19.4s, #0x0\n"
        "ldr q27, [x25, #0x0]\n"
        "ldr q15, [x25, #0x10]\n"
        "movi v26.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        "ldr q29, [x28, #0x20]\n"
        "ldr q3, [x28, #0x30]\n"
        "movi v17.4s, #0x0\n"
        "movi v0.16b, #0xf0\n"
        "ldr d20, [x25, #-0x8]\n"
        "ldr d9, [x23, #-0x8]\n"
        "sshl v8.16b, v21.16b, v1.16b\n"
        "sshl v31.16b, v16.16b, v1.16b\n"
        "and v21.16b, v21.16b, v0.16b\n"
        "and v16.16b, v16.16b, v0.16b\n"
        "sub x20, x28, #0x8\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
        ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
        "ldr q27, [x25, #0x20]\n"
        ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
        ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
        "sshl v15.16b, v29.16b, v1.16b\n"
        "sshl v1.16b, v3.16b, v1.16b\n"
        "and v29.16b, v29.16b, v0.16b\n"
        "and v3.16b, v3.16b, v0.16b\n"
        "ldr q0, [x25, #0x30]\n"
        "fcvtl v20.4s, v20.4h\n"
        ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
        "fcvtl v9.4s, v9.4h\n"
        ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
        "ldr q27, [x25, #0x40]\n"
        ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
        ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
        "ldr q0, [x25, #0x50]\n"
        ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
        ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
        "ldr q27, [x25, #0x60]\n"
        ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
        ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
        "ldr q0, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
        ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
        "ldr d27, [x20, #0x0]\n"
        ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
        ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
        "fcvtl v27.4s, v27.4h\n"
        "uzp1 v0.2d, v19.2d, v26.2d\n"
        "uzp2 v26.2d, v19.2d, v26.2d\n"
        "fmul v19.4s, v27.4s, v20.s[0]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "fmla v2.4s, v0.4s, v19.4s\n"
        "ldr q19, [x23, #0x0]\n"
        "uzp1 v0.2d, v18.2d, v17.2d\n"
        "uzp2 v18.2d, v18.2d, v17.2d\n"
        "fmul v17.4s, v27.4s, v20.s[1]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v10.4s, v26.4s, v17.4s\n"
        "ldr q17, [x23, #0x10]\n"
        "fmul v26.4s, v27.4s, v20.s[2]\n"
        "fmul v20.4s, v27.4s, v20.s[3]\n"
        "fmla v12.4s, v0.4s, v26.4s\n"
        "ldr d0, [x22, #-0x8]\n"
        "ldr d26, [x21, #-0x8]\n"
        "fcvtl v0.4s, v0.4h\n"
        "fmla v28.4s, v18.4s, v20.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x23, #0x20]\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x23, #0x40]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q19, [x23, #0x60]\n"
        ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
        ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
        "uzp1 v19.2d, v20.2d, v18.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp2 v20.2d, v20.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v9.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v11.4s, v19.4s, v18.4s\n"
        "ldr q18, [x22, #0x0]\n"
        "fmul v19.4s, v27.4s, v9.s[1]\n"
        "fmla v13.4s, v20.4s, v19.4s\n"
        "movi v19.4s, #0x0\n"
        "movi v20.4s, #0x0\n"
        ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
        ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
        "ldr q17, [x23, #0x30]\n"
        ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
        "ldr q17, [x23, #0x50]\n"
        ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
        "ldr q17, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v9.s[2]\n"
        "fmul v9.4s, v27.4s, v9.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v22.4s, v17.4s, v19.4s\n"
        "ldr q17, [x22, #0x10]\n"
        "movi v19.4s, #0x0\n"
        ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
        "fmla v23.4s, v20.4s, v9.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
        "ldr q18, [x22, #0x20]\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
        ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
        "ldr q18, [x22, #0x40]\n"
        ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
        ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
        "ldr q18, [x22, #0x60]\n"
        ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
        ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
        "ldr q17, [x22, #0x30]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
        "ldr q17, [x22, #0x50]\n"
        ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
        "ldr q17, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v0.s[0]\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v25.4s, v17.4s, v19.4s\n"
        "ldr q19, [x21, #0x0]\n"
        "fmul v17.4s, v27.4s, v0.s[1]\n"
        "fmla v5.4s, v20.4s, v17.4s\n"
        "ldr q17, [x21, #0x10]\n"
        "uzp1 v20.2d, v9.2d, v18.2d\n"
        "uzp2 v9.2d, v9.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v0.s[2]\n"
        "fmul v0.4s, v27.4s, v0.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "fmla v7.4s, v20.4s, v18.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x21, #0x20]\n"
        "fmla v4.4s, v9.4s, v0.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        "fmul v8.4s, v27.4s, v26.s[0]\n"
        ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
        "ldr q17, [x21, #0x30]\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        "fmul v31.4s, v27.4s, v26.s[1]\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x21, #0x40]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        "fmul v15.4s, v27.4s, v26.s[2]\n"
        "fmul v27.4s, v27.4s, v26.s[3]\n"
        ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
        "ldr q1, [x21, #0x50]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q26, [x21, #0x60]\n"
        ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
        ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
        "ldr q21, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
        ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
        ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
        ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
        "uzp1 v29.2d, v20.2d, v18.2d\n"
        "uzp2 v21.2d, v20.2d, v18.2d\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "uzp1 v18.2d, v9.2d, v0.2d\n"
        "uzp2 v16.2d, v9.2d, v0.2d\n"
        "scvtf v21.4s, v21.4s, #0x4\n"
        "fmla v6.4s, v29.4s, v8.4s\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v30.4s, v21.4s, v31.4s\n"
        "fmla v24.4s, v18.4s, v15.4s\n"
        "fmla v14.4s, v16.4s, v27.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q28, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q6, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q24, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:" // Row loop skip
        "cbz x10, 9f\n"
        "5:" // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[nc]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:" // Row tail: Column loop
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[nb]\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "7:" // Row tail: Block loop
        "ldr q6, [x24, #0x0]\n"
        "ldr q5, [x24, #0x10]\n"
        "movi v17.16b, #0x4\n"
        "movi v8.4s, #0x0\n"
        "ldr q4, [x25, #0x0]\n"
        "ldr q13, [x25, #0x10]\n"
        "movi v27.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q31, [x24, #0x20]\n"
        "ldr q14, [x24, #0x30]\n"
        "movi v29.4s, #0x0\n"
        "movi v22.16b, #0xf0\n"
        "ldr q11, [x25, #0x20]\n"
        "ldr q23, [x25, #0x30]\n"
        "sshl v21.16b, v6.16b, v17.16b\n"
        "sshl v16.16b, v5.16b, v17.16b\n"
        "ldr q20, [x25, #0x40]\n"
        "ldr q26, [x25, #0x50]\n"
        "and v6.16b, v6.16b, v22.16b\n"
        "and v5.16b, v5.16b, v22.16b\n"
        "ldr q25, [x25, #0x60]\n"
        "ldr q3, [x25, #0x70]\n"
        "sshl v19.16b, v31.16b, v17.16b\n"
        "sshl v18.16b, v14.16b, v17.16b\n"
        "ldr d17, [x25, #-0x8]\n"
        ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
        ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
        "and v31.16b, v31.16b, v22.16b\n"
        ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
        ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
        "and v14.16b, v14.16b, v22.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr d16, [x20, #0x0]\n"
        "subs x21, x21, #0x1\n"
        "add x25, x25, #0x88\n"
        "fcvtl v17.4s, v17.4h\n"
        "add x24, x24, #0x48\n"
        ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
        ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
        ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
        ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
        "fcvtl v16.4s, v16.4h\n"
        ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
        ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
        "fmul v23.4s, v16.4s, v17.s[0]\n"
        "fmul v21.4s, v16.4s, v17.s[1]\n"
        "fmul v1.4s, v16.4s, v17.s[2]\n"
        "fmul v20.4s, v16.4s, v17.s[3]\n"
        ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
        ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
        ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
        ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
        ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
        ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
        "uzp1 v19.2d, v8.2d, v27.2d\n"
        "uzp2 v18.2d, v8.2d, v27.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp1 v17.2d, v0.2d, v29.2d\n"
        "uzp2 v16.2d, v0.2d, v29.2d\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v2.4s, v19.4s, v23.4s\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v10.4s, v18.4s, v21.4s\n"
        "fmla v12.4s, v17.4s, v1.4s\n"
        "fmla v28.4s, v16.4s, v20.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q28, [x20, #0x0]\n"
        "8:" // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:" // Row tail: Row loop skip
        : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
        : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb), [res_stride] "r"(res_stride), [nc] "r"(nc)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
          "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
          "x24", "x25", "x26", "x27", "x28");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    assert((mllm_cpu_has_sve() || mllm_cpu_has_matmul_int8())
           && "__ARM_FEATURE_SVE and __ARM_FEATURE_MATMUL_INT8 not defined, use the "
              "Q4_0_4_4 quantization format for optimal "
              "performance");
#else
    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        << 4);
                                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i])
                                         + (v1
                                            * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i
                                                          + qk / 2 * 4]))
                                        >> 4;
                            }
                            sumf[m][j] += sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j])
                                          * MLLM_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
#endif
}

void _gemm_q4_0_4x8_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc,
                              const void *__restrict bias) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

// #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
//     if (svcntw() == 8) {
//         assert(!(mllm_cpu_has_sve() && (svcntw() == 8))
//                && "__ARM_FEATURE_SVE defined, use the Q4_0_8_8 quantization format "
//                   "for optimal performance");
//     }
// #endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    const void *bias_ptr = bias;
    float *res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[nb], x9\n"
        "blt 4f\n"
        "1:" // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[nc]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:" // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "mov x24, %x[nb]\n"
        "add x23, x25, x9\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v22.16b, #0x0\n"
        "movi v23.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v6.16b, #0x0\n"
        "movi v30.16b, #0x0\n"
        "movi v24.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "3:" // Block loop
        "ldr q21, [x28, #0x0]\n"
        "ldr q16, [x28, #0x10]\n"
        "movi v1.16b, #0x4\n"
        "movi v19.4s, #0x0\n"
        "ldr q27, [x25, #0x0]\n"
        "ldr q15, [x25, #0x10]\n"
        "movi v26.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        "ldr q29, [x28, #0x20]\n"
        "ldr q3, [x28, #0x30]\n"
        "movi v17.4s, #0x0\n"
        "movi v0.16b, #0xf0\n"
        "ldr d20, [x25, #-0x8]\n"
        "ldr d9, [x23, #-0x8]\n"
        "sshl v8.16b, v21.16b, v1.16b\n"
        "sshl v31.16b, v16.16b, v1.16b\n"
        "and v21.16b, v21.16b, v0.16b\n"
        "and v16.16b, v16.16b, v0.16b\n"
        "sub x20, x28, #0x8\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
        ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
        "ldr q27, [x25, #0x20]\n"
        ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
        ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
        "sshl v15.16b, v29.16b, v1.16b\n"
        "sshl v1.16b, v3.16b, v1.16b\n"
        "and v29.16b, v29.16b, v0.16b\n"
        "and v3.16b, v3.16b, v0.16b\n"
        "ldr q0, [x25, #0x30]\n"
        "fcvtl v20.4s, v20.4h\n"
        ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
        "fcvtl v9.4s, v9.4h\n"
        ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
        "ldr q27, [x25, #0x40]\n"
        ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
        ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
        "ldr q0, [x25, #0x50]\n"
        ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
        ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
        "ldr q27, [x25, #0x60]\n"
        ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
        ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
        "ldr q0, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
        ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
        "ldr d27, [x20, #0x0]\n"
        ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
        ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
        "fcvtl v27.4s, v27.4h\n"
        "uzp1 v0.2d, v19.2d, v26.2d\n"
        "uzp2 v26.2d, v19.2d, v26.2d\n"
        "fmul v19.4s, v27.4s, v20.s[0]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "fmla v2.4s, v0.4s, v19.4s\n"
        "ldr q19, [x23, #0x0]\n"
        "uzp1 v0.2d, v18.2d, v17.2d\n"
        "uzp2 v18.2d, v18.2d, v17.2d\n"
        "fmul v17.4s, v27.4s, v20.s[1]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v10.4s, v26.4s, v17.4s\n"
        "ldr q17, [x23, #0x10]\n"
        "fmul v26.4s, v27.4s, v20.s[2]\n"
        "fmul v20.4s, v27.4s, v20.s[3]\n"
        "fmla v12.4s, v0.4s, v26.4s\n"
        "ldr d0, [x22, #-0x8]\n"
        "ldr d26, [x21, #-0x8]\n"
        "fcvtl v0.4s, v0.4h\n"
        "fmla v28.4s, v18.4s, v20.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x23, #0x20]\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x23, #0x40]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q19, [x23, #0x60]\n"
        ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
        ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
        "uzp1 v19.2d, v20.2d, v18.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp2 v20.2d, v20.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v9.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v11.4s, v19.4s, v18.4s\n"
        "ldr q18, [x22, #0x0]\n"
        "fmul v19.4s, v27.4s, v9.s[1]\n"
        "fmla v13.4s, v20.4s, v19.4s\n"
        "movi v19.4s, #0x0\n"
        "movi v20.4s, #0x0\n"
        ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
        ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
        "ldr q17, [x23, #0x30]\n"
        ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
        "ldr q17, [x23, #0x50]\n"
        ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
        "ldr q17, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v9.s[2]\n"
        "fmul v9.4s, v27.4s, v9.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v22.4s, v17.4s, v19.4s\n"
        "ldr q17, [x22, #0x10]\n"
        "movi v19.4s, #0x0\n"
        ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
        "fmla v23.4s, v20.4s, v9.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
        "ldr q18, [x22, #0x20]\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
        ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
        "ldr q18, [x22, #0x40]\n"
        ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
        ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
        "ldr q18, [x22, #0x60]\n"
        ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
        ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
        "ldr q17, [x22, #0x30]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
        "ldr q17, [x22, #0x50]\n"
        ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
        "ldr q17, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v0.s[0]\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v25.4s, v17.4s, v19.4s\n"
        "ldr q19, [x21, #0x0]\n"
        "fmul v17.4s, v27.4s, v0.s[1]\n"
        "fmla v5.4s, v20.4s, v17.4s\n"
        "ldr q17, [x21, #0x10]\n"
        "uzp1 v20.2d, v9.2d, v18.2d\n"
        "uzp2 v9.2d, v9.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v0.s[2]\n"
        "fmul v0.4s, v27.4s, v0.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "fmla v7.4s, v20.4s, v18.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x21, #0x20]\n"
        "fmla v4.4s, v9.4s, v0.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        "fmul v8.4s, v27.4s, v26.s[0]\n"
        ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
        "ldr q17, [x21, #0x30]\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        "fmul v31.4s, v27.4s, v26.s[1]\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x21, #0x40]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        "fmul v15.4s, v27.4s, v26.s[2]\n"
        "fmul v27.4s, v27.4s, v26.s[3]\n"
        ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
        "ldr q1, [x21, #0x50]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q26, [x21, #0x60]\n"
        ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
        ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
        "ldr q21, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
        ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
        ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
        ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
        "uzp1 v29.2d, v20.2d, v18.2d\n"
        "uzp2 v21.2d, v20.2d, v18.2d\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "uzp1 v18.2d, v9.2d, v0.2d\n"
        "uzp2 v16.2d, v9.2d, v0.2d\n"
        "scvtf v21.4s, v21.4s, #0x4\n"
        "fmla v6.4s, v29.4s, v8.4s\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v30.4s, v21.4s, v31.4s\n"
        "fmla v24.4s, v18.4s, v15.4s\n"
        "fmla v14.4s, v16.4s, v27.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q28, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q6, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q24, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:" // Row loop skip
        "cbz x10, 9f\n"
        "5:" // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[nc]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:" // Row tail: Column loop
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[nb]\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "7:" // Row tail: Block loop
        "ldr q6, [x24, #0x0]\n"
        "ldr q5, [x24, #0x10]\n"
        "movi v17.16b, #0x4\n"
        "movi v8.4s, #0x0\n"
        "ldr q4, [x25, #0x0]\n"
        "ldr q13, [x25, #0x10]\n"
        "movi v27.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q31, [x24, #0x20]\n"
        "ldr q14, [x24, #0x30]\n"
        "movi v29.4s, #0x0\n"
        "movi v22.16b, #0xf0\n"
        "ldr q11, [x25, #0x20]\n"
        "ldr q23, [x25, #0x30]\n"
        "sshl v21.16b, v6.16b, v17.16b\n"
        "sshl v16.16b, v5.16b, v17.16b\n"
        "ldr q20, [x25, #0x40]\n"
        "ldr q26, [x25, #0x50]\n"
        "and v6.16b, v6.16b, v22.16b\n"
        "and v5.16b, v5.16b, v22.16b\n"
        "ldr q25, [x25, #0x60]\n"
        "ldr q3, [x25, #0x70]\n"
        "sshl v19.16b, v31.16b, v17.16b\n"
        "sshl v18.16b, v14.16b, v17.16b\n"
        "ldr d17, [x25, #-0x8]\n"
        ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
        ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
        "and v31.16b, v31.16b, v22.16b\n"
        ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
        ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
        "and v14.16b, v14.16b, v22.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr d16, [x20, #0x0]\n"
        "subs x21, x21, #0x1\n"
        "add x25, x25, #0x88\n"
        "fcvtl v17.4s, v17.4h\n"
        "add x24, x24, #0x48\n"
        ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
        ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
        ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
        ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
        "fcvtl v16.4s, v16.4h\n"
        ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
        ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
        "fmul v23.4s, v16.4s, v17.s[0]\n"
        "fmul v21.4s, v16.4s, v17.s[1]\n"
        "fmul v1.4s, v16.4s, v17.s[2]\n"
        "fmul v20.4s, v16.4s, v17.s[3]\n"
        ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
        ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
        ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
        ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
        ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
        ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
        "uzp1 v19.2d, v8.2d, v27.2d\n"
        "uzp2 v18.2d, v8.2d, v27.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp1 v17.2d, v0.2d, v29.2d\n"
        "uzp2 v16.2d, v0.2d, v29.2d\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v2.4s, v19.4s, v23.4s\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v10.4s, v18.4s, v21.4s\n"
        "fmla v12.4s, v17.4s, v1.4s\n"
        "fmla v28.4s, v16.4s, v20.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q28, [x20, #0x0]\n"
        "8:" // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:" // Row tail: Row loop skip
        : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
        : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb), [res_stride] "r"(res_stride), [nc] "r"(nc)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
          "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
          "x24", "x25", "x26", "x27", "x28");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    assert((mllm_cpu_has_sve() || mllm_cpu_has_matmul_int8())
           && "__ARM_FEATURE_SVE and __ARM_FEATURE_MATMUL_INT8 not defined, use the "
              "Q4_0_4_4 quantization format for optimal "
              "performance");
#else
    float sumf[4][4];
    int sumi;

    const float *bias_ptr = (const float *)bias;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    sumf[m][j] = bias_ptr[x * ncols_interleaved + j];
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        << 4);
                                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i])
                                         + (v1
                                            * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i
                                                          + qk / 2 * 4]))
                                        >> 4;
                            }
                            sumf[m][j] += sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j])
                                          * MLLM_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
#endif
}

void gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                        const void *__restrict vy, int nr, int nc,
                        const void *__restrict bias) {
    if (bias != nullptr) {
#if defined(__ARM_NEON)
        std::cout << "_gemm_q4_0_8x8_q8_0_bias not implemented";
        abort();
#endif
        _gemm_q4_0_8x8_q8_0_bias(n, s, bs, vx, vy, nr, nc, bias);
        return;
    }

    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    // if (svcntw() == 8) {
    if (true) {
        const void *b_ptr = vx;
        const void *a_ptr = vy;
        float *res_ptr = s;
        size_t res_stride = bs * sizeof(float);

        __asm__ __volatile__("mov x20, #0x4\n"
                             "mov x13, %x[nr]\n"
                             "mov z28.s, #-0x4\n"
                             "mov x12, #0x88\n"
                             "ptrue p1.b\n"
                             "whilelt p0.s, XZR, x20\n"
                             "cmp x13, #0x10\n"
                             "mul x12, %x[nb], x12\n"
                             "blt 4f\n"
                             "1:" // Row loop
                             "add x11, %x[b_ptr], #0x10\n"
                             "mov x10, %x[nc]\n"
                             "add x9, %x[res_ptr], %x[res_stride], LSL #4\n"
                             "2:" // Column loop
                             "add x28, %x[a_ptr], #0x8\n"
                             "mov z24.b, #0x0\n"
                             "mov z15.b, #0x0\n"
                             "mov x27, %x[nb]\n"
                             "add x26, x28, x12\n"
                             "mov z12.b, #0x0\n"
                             "mov z0.b, #0x0\n"
                             "add x25, x26, x12\n"
                             "mov z13.b, #0x0\n"
                             "mov z1.b, #0x0\n"
                             "add x24, x25, x12\n"
                             "mov z20.b, #0x0\n"
                             "mov z25.b, #0x0\n"
                             "mov z11.b, #0x0\n"
                             "mov z16.b, #0x0\n"
                             "mov z19.b, #0x0\n"
                             "mov z26.b, #0x0\n"
                             "mov z8.b, #0x0\n"
                             "mov z29.b, #0x0\n"
                             "mov z27.b, #0x0\n"
                             "mov z10.b, #0x0\n"
                             "3:" // Block loop
                             "ld1b { z30.b }, p1/Z, [x11]\n"
                             "ld1b { z21.b }, p1/Z, [x11, #1, MUL VL]\n"
                             "mov z18.s, #0x0\n"
                             "mov z7.s, #0x0\n"
                             "ld1rqb { z3.b }, p1/Z, [x28]\n"
                             "ld1rqb { z5.b }, p1/Z, [x28, #16]\n"
                             "mov z9.s, #0x0\n"
                             "mov z22.s, #0x0\n"
                             "ld1b { z4.b }, p1/Z, [x11, #2, MUL VL]\n"
                             "ld1b { z17.b }, p1/Z, [x11, #3, MUL VL]\n"
                             "sub x20, x11, #0x10\n"
                             "sub x23, x28, #0x8\n"
                             "lsl z31.b, z30.b, #0x4\n"
                             "lsl z6.b, z21.b, #0x4\n"
                             "ld1h { z23.s }, p1/Z, [x20]\n"
                             "sub x22, x26, #0x8\n"
                             "and z30.b, z30.b, #0xf0\n"
                             "and z21.b, z21.b, #0xf0\n"
                             "sub x21, x25, #0x8\n"
                             "sub x20, x24, #0x8\n"
                             "lsl z14.b, z4.b, #0x4\n"
                             "lsl z2.b, z17.b, #0x4\n"
                             "subs x27, x27, #0x1\n"
                             "add x11, x11, #0x90\n"
                             ".inst 0x451f9872  // smmla z18.s, z3.b, z31.b\n"
                             ".inst 0x45069867  // smmla z7.s, z3.b, z6.b\n"
                             "ld1rqb { z3.b }, p1/Z, [x28, #32]\n"
                             "and z4.b, z4.b, #0xf0\n"
                             ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
                             ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x28, #48]\n"
                             "and z17.b, z17.b, #0xf0\n"
                             "fcvt z23.s, p1/m, z23.h\n"
                             ".inst 0x450e9872  // smmla z18.s, z3.b, z14.b\n"
                             ".inst 0x45029867  // smmla z7.s, z3.b, z2.b\n"
                             "ld1rqb { z3.b }, p1/Z, [x28, #64]\n"
                             ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
                             ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x28, #80]\n"
                             "fscale z23.s, p1/m, z23.s, z28.s\n"
                             ".inst 0x451e9872  // smmla z18.s, z3.b, z30.b\n"
                             ".inst 0x45159867  // smmla z7.s, z3.b, z21.b\n"
                             "ld1rqb { z3.b }, p1/Z, [x28, #96]\n"
                             ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
                             ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x28, #112]\n"
                             "add x28, x28, #0x88\n"
                             ".inst 0x45049872  // smmla z18.s, z3.b, z4.b\n"
                             ".inst 0x45119867  // smmla z7.s, z3.b, z17.b\n"
                             "ld1h { z3.s }, p0/Z, [x23]\n"
                             ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
                             ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
                             "fcvt z3.s, p1/m, z3.h\n"
                             "uzp1 z5.d, z18.d, z7.d\n"
                             "uzp2 z18.d, z18.d, z7.d\n"
                             "mov z3.q, z3.q[0]\n"
                             "uzp1 z7.d, z9.d, z22.d\n"
                             "uzp2 z22.d, z9.d, z22.d\n"
                             "fmul z9.s, z23.s, z3.s[0]\n"
                             "scvtf z5.s, p1/m, z5.s\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "scvtf z7.s, p1/m, z7.s\n"
                             "scvtf z22.s, p1/m, z22.s\n"
                             "fmla z24.s, p1/M, z5.s, z9.s\n"
                             "ld1rqb { z5.b }, p1/Z, [x26]\n"
                             "fmul z9.s, z23.s, z3.s[1]\n"
                             "fmla z15.s, p1/M, z18.s, z9.s\n"
                             "ld1rqb { z18.b }, p1/Z, [x26, #16]\n"
                             "fmul z9.s, z23.s, z3.s[2]\n"
                             "fmul z3.s, z23.s, z3.s[3]\n"
                             "fmla z12.s, p1/M, z7.s, z9.s\n"
                             "mov z9.s, #0x0\n"
                             "ld1h { z7.s }, p0/Z, [x22]\n"
                             ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
                             "fmla z0.s, p1/M, z22.s, z3.s\n"
                             "mov z22.s, #0x0\n"
                             "ld1h { z3.s }, p0/Z, [x21]\n"
                             ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x26, #32]\n"
                             "fcvt z7.s, p1/m, z7.h\n"
                             "fcvt z3.s, p1/m, z3.h\n"
                             ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
                             ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x26, #64]\n"
                             "mov z7.q, z7.q[0]\n"
                             "mov z3.q, z3.q[0]\n"
                             ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
                             ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x26, #96]\n"
                             ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
                             ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
                             "uzp1 z5.d, z9.d, z22.d\n"
                             "scvtf z5.s, p1/m, z5.s\n"
                             "uzp2 z22.d, z9.d, z22.d\n"
                             "fmul z9.s, z23.s, z7.s[0]\n"
                             "scvtf z22.s, p1/m, z22.s\n"
                             "fmla z13.s, p1/M, z5.s, z9.s\n"
                             "ld1rqb { z9.b }, p1/Z, [x25]\n"
                             "fmul z5.s, z23.s, z7.s[1]\n"
                             "fmla z1.s, p1/M, z22.s, z5.s\n"
                             "mov z5.s, #0x0\n"
                             "mov z22.s, #0x0\n"
                             ".inst 0x451f9a45  // smmla z5.s, z18.b, z31.b\n"
                             ".inst 0x45069a56  // smmla z22.s, z18.b, z6.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x26, #48]\n"
                             ".inst 0x450e9a45  // smmla z5.s, z18.b, z14.b\n"
                             ".inst 0x45029a56  // smmla z22.s, z18.b, z2.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x26, #80]\n"
                             ".inst 0x451e9a45  // smmla z5.s, z18.b, z30.b\n"
                             ".inst 0x45159a56  // smmla z22.s, z18.b, z21.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x26, #112]\n"
                             "add x26, x26, #0x88\n"
                             ".inst 0x45049a45  // smmla z5.s, z18.b, z4.b\n"
                             ".inst 0x45119a56  // smmla z22.s, z18.b, z17.b\n"
                             "uzp1 z18.d, z5.d, z22.d\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "uzp2 z22.d, z5.d, z22.d\n"
                             "fmul z5.s, z23.s, z7.s[2]\n"
                             "fmul z7.s, z23.s, z7.s[3]\n"
                             "scvtf z22.s, p1/m, z22.s\n"
                             "fmla z20.s, p1/M, z18.s, z5.s\n"
                             "ld1rqb { z18.b }, p1/Z, [x25, #16]\n"
                             "ld1h { z5.s }, p0/Z, [x20]\n"
                             "fcvt z5.s, p1/m, z5.h\n"
                             "fmla z25.s, p1/M, z22.s, z7.s\n"
                             "mov z22.s, #0x0\n"
                             "mov z7.s, #0x0\n"
                             ".inst 0x451f9936  // smmla z22.s, z9.b, z31.b\n"
                             ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
                             "ld1rqb { z9.b }, p1/Z, [x25, #32]\n"
                             "mov z5.q, z5.q[0]\n"
                             ".inst 0x450e9936  // smmla z22.s, z9.b, z14.b\n"
                             ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
                             "ld1rqb { z9.b }, p1/Z, [x25, #64]\n"
                             ".inst 0x451e9936  // smmla z22.s, z9.b, z30.b\n"
                             ".inst 0x45159927  // smmla z7.s, z9.b, z21.b\n"
                             "ld1rqb { z9.b }, p1/Z, [x25, #96]\n"
                             ".inst 0x45049936  // smmla z22.s, z9.b, z4.b\n"
                             ".inst 0x45119927  // smmla z7.s, z9.b, z17.b\n"
                             "uzp1 z9.d, z22.d, z7.d\n"
                             "scvtf z9.s, p1/m, z9.s\n"
                             "uzp2 z22.d, z22.d, z7.d\n"
                             "fmul z7.s, z23.s, z3.s[0]\n"
                             "scvtf z22.s, p1/m, z22.s\n"
                             "fmla z11.s, p1/M, z9.s, z7.s\n"
                             "ld1rqb { z9.b }, p1/Z, [x24]\n"
                             "fmul z7.s, z23.s, z3.s[1]\n"
                             "fmla z16.s, p1/M, z22.s, z7.s\n"
                             "mov z22.s, #0x0\n"
                             "mov z7.s, #0x0\n"
                             ".inst 0x451f9a56  // smmla z22.s, z18.b, z31.b\n"
                             ".inst 0x45069a47  // smmla z7.s, z18.b, z6.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x25, #48]\n"
                             ".inst 0x450e9a56  // smmla z22.s, z18.b, z14.b\n"
                             ".inst 0x45029a47  // smmla z7.s, z18.b, z2.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x25, #80]\n"
                             ".inst 0x451e9a56  // smmla z22.s, z18.b, z30.b\n"
                             ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x25, #112]\n"
                             "add x25, x25, #0x88\n"
                             ".inst 0x45049a56  // smmla z22.s, z18.b, z4.b\n"
                             ".inst 0x45119a47  // smmla z7.s, z18.b, z17.b\n"
                             "uzp1 z18.d, z22.d, z7.d\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "uzp2 z7.d, z22.d, z7.d\n"
                             "fmul z22.s, z23.s, z3.s[2]\n"
                             "fmul z3.s, z23.s, z3.s[3]\n"
                             "scvtf z7.s, p1/m, z7.s\n"
                             "fmla z19.s, p1/M, z18.s, z22.s\n"
                             "ld1rqb { z18.b }, p1/Z, [x24, #16]\n"
                             "fmul z22.s, z23.s, z5.s[0]\n"
                             "fmla z26.s, p1/M, z7.s, z3.s\n"
                             "mov z3.s, #0x0\n"
                             "mov z7.s, #0x0\n"
                             ".inst 0x451f9923  // smmla z3.s, z9.b, z31.b\n"
                             ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
                             "ld1rqb { z9.b }, p1/Z, [x24, #32]\n"
                             ".inst 0x450e9923  // smmla z3.s, z9.b, z14.b\n"
                             ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
                             "mov z9.s, #0x0\n"
                             ".inst 0x451f9a49  // smmla z9.s, z18.b, z31.b\n"
                             "mov z31.s, #0x0\n"
                             ".inst 0x45069a5f  // smmla z31.s, z18.b, z6.b\n"
                             "ld1rqb { z6.b }, p1/Z, [x24, #48]\n"
                             "ld1rqb { z18.b }, p1/Z, [x24, #64]\n"
                             ".inst 0x450e98c9  // smmla z9.s, z6.b, z14.b\n"
                             "fmul z14.s, z23.s, z5.s[1]\n"
                             ".inst 0x450298df  // smmla z31.s, z6.b, z2.b\n"
                             "ld1rqb { z6.b }, p1/Z, [x24, #80]\n"
                             "fmul z2.s, z23.s, z5.s[2]\n"
                             "fmul z23.s, z23.s, z5.s[3]\n"
                             ".inst 0x451e9a43  // smmla z3.s, z18.b, z30.b\n"
                             ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x24, #96]\n"
                             ".inst 0x451e98c9  // smmla z9.s, z6.b, z30.b\n"
                             ".inst 0x451598df  // smmla z31.s, z6.b, z21.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x24, #112]\n"
                             "add x24, x24, #0x88\n"
                             ".inst 0x450498a3  // smmla z3.s, z5.b, z4.b\n"
                             ".inst 0x451198a7  // smmla z7.s, z5.b, z17.b\n"
                             ".inst 0x45049a49  // smmla z9.s, z18.b, z4.b\n"
                             ".inst 0x45119a5f  // smmla z31.s, z18.b, z17.b\n"
                             "uzp1 z18.d, z3.d, z7.d\n"
                             "uzp2 z5.d, z3.d, z7.d\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "uzp1 z6.d, z9.d, z31.d\n"
                             "uzp2 z9.d, z9.d, z31.d\n"
                             "scvtf z5.s, p1/m, z5.s\n"
                             "fmla z8.s, p1/M, z18.s, z22.s\n"
                             "scvtf z6.s, p1/m, z6.s\n"
                             "scvtf z9.s, p1/m, z9.s\n"
                             "fmla z29.s, p1/M, z5.s, z14.s\n"
                             "fmla z27.s, p1/M, z6.s, z2.s\n"
                             "fmla z10.s, p1/M, z9.s, z23.s\n"
                             "bgt 3b\n"
                             "mov x20, %x[res_ptr]\n"
                             "subs x10, x10, #0x8\n"
                             "add %x[res_ptr], %x[res_ptr], #0x20\n"
                             "st1w { z24.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z15.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z12.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z0.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z13.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z1.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z20.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z25.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z11.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z16.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z19.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z26.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z8.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z29.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z27.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z10.s }, p1, [x20]\n"
                             "bne 2b\n"
                             "mov x20, #0x4\n"
                             "sub x13, x13, #0x10\n"
                             "cmp x13, #0x10\n"
                             "mov %x[res_ptr], x9\n"
                             "madd %x[a_ptr], x20, x12, %x[a_ptr]\n"
                             "bge 1b\n"
                             "4:" // Row loop skip
                             "cbz x13, 9f\n"
                             "5:" // Row tail: Row loop
                             "add x25, %x[b_ptr], #0x10\n"
                             "mov x24, %x[nc]\n"
                             "add x23, %x[res_ptr], %x[res_stride], LSL #2\n"
                             "6:" // Row tail: Column loop
                             "mov z24.b, #0x0\n"
                             "mov z15.b, #0x0\n"
                             "add x28, %x[a_ptr], #0x8\n"
                             "mov x22, %x[nb]\n"
                             "mov z12.b, #0x0\n"
                             "mov z0.b, #0x0\n"
                             "7:" // Row tail: Block loop
                             "ld1b { z3.b }, p1/Z, [x25]\n"
                             "ld1b { z6.b }, p1/Z, [x25, #1, MUL VL]\n"
                             "mov z2.s, #0x0\n"
                             "mov z25.s, #0x0\n"
                             "ld1rqb { z26.b }, p1/Z, [x28]\n"
                             "ld1rqb { z21.b }, p1/Z, [x28, #16]\n"
                             "mov z27.s, #0x0\n"
                             "mov z19.s, #0x0\n"
                             "ld1b { z29.b }, p1/Z, [x25, #2, MUL VL]\n"
                             "ld1b { z16.b }, p1/Z, [x25, #3, MUL VL]\n"
                             "sub x21, x25, #0x10\n"
                             "sub x20, x28, #0x8\n"
                             "lsl z20.b, z3.b, #0x4\n"
                             "lsl z4.b, z6.b, #0x4\n"
                             "ld1rqb { z10.b }, p1/Z, [x28, #32]\n"
                             "ld1rqb { z23.b }, p1/Z, [x28, #48]\n"
                             "and z3.b, z3.b, #0xf0\n"
                             "and z6.b, z6.b, #0xf0\n"
                             "ld1rqb { z11.b }, p1/Z, [x28, #64]\n"
                             "ld1rqb { z7.b }, p1/Z, [x28, #80]\n"
                             "lsl z8.b, z29.b, #0x4\n"
                             "lsl z14.b, z16.b, #0x4\n"
                             "ld1rqb { z18.b }, p1/Z, [x28, #96]\n"
                             "ld1rqb { z30.b }, p1/Z, [x28, #112]\n"
                             ".inst 0x45149b42  // smmla z2.s, z26.b, z20.b\n"
                             ".inst 0x45049b59  // smmla z25.s, z26.b, z4.b\n"
                             "and z29.b, z29.b, #0xf0\n"
                             "ld1h { z17.s }, p1/Z, [x21]\n"
                             ".inst 0x45149abb  // smmla z27.s, z21.b, z20.b\n"
                             ".inst 0x45049ab3  // smmla z19.s, z21.b, z4.b\n"
                             "and z16.b, z16.b, #0xf0\n"
                             "ld1h { z4.s }, p0/Z, [x20]\n"
                             "subs x22, x22, #0x1\n"
                             "add x28, x28, #0x88\n"
                             "fcvt z17.s, p1/m, z17.h\n"
                             "add x25, x25, #0x90\n"
                             ".inst 0x45089942  // smmla z2.s, z10.b, z8.b\n"
                             ".inst 0x450e9959  // smmla z25.s, z10.b, z14.b\n"
                             "fcvt z4.s, p1/m, z4.h\n"
                             ".inst 0x45089afb  // smmla z27.s, z23.b, z8.b\n"
                             ".inst 0x450e9af3  // smmla z19.s, z23.b, z14.b\n"
                             "fscale z17.s, p1/m, z17.s, z28.s\n"
                             "mov z4.q, z4.q[0]\n"
                             ".inst 0x45039962  // smmla z2.s, z11.b, z3.b\n"
                             ".inst 0x45069979  // smmla z25.s, z11.b, z6.b\n"
                             "fmul z23.s, z17.s, z4.s[0]\n"
                             "fmul z9.s, z17.s, z4.s[1]\n"
                             "fmul z21.s, z17.s, z4.s[2]\n"
                             "fmul z4.s, z17.s, z4.s[3]\n"
                             ".inst 0x450398fb  // smmla z27.s, z7.b, z3.b\n"
                             ".inst 0x450698f3  // smmla z19.s, z7.b, z6.b\n"
                             ".inst 0x451d9a42  // smmla z2.s, z18.b, z29.b\n"
                             ".inst 0x45109a59  // smmla z25.s, z18.b, z16.b\n"
                             ".inst 0x451d9bdb  // smmla z27.s, z30.b, z29.b\n"
                             ".inst 0x45109bd3  // smmla z19.s, z30.b, z16.b\n"
                             "uzp1 z31.d, z2.d, z25.d\n"
                             "uzp2 z13.d, z2.d, z25.d\n"
                             "scvtf z31.s, p1/m, z31.s\n"
                             "uzp1 z17.d, z27.d, z19.d\n"
                             "uzp2 z18.d, z27.d, z19.d\n"
                             "scvtf z13.s, p1/m, z13.s\n"
                             "fmla z24.s, p1/M, z31.s, z23.s\n"
                             "scvtf z17.s, p1/m, z17.s\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "fmla z15.s, p1/M, z13.s, z9.s\n"
                             "fmla z12.s, p1/M, z17.s, z21.s\n"
                             "fmla z0.s, p1/M, z18.s, z4.s\n"
                             "bgt 7b\n"
                             "mov x20, %x[res_ptr]\n"
                             "cmp x13, #0x1\n"
                             "st1w { z24.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "ble 8f\n"
                             "cmp x13, #0x2\n"
                             "st1w { z15.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "ble 8f\n"
                             "cmp x13, #0x3\n"
                             "st1w { z12.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "ble 8f\n"
                             "st1w { z0.s }, p1, [x20]\n"
                             "8:" // Row tail: Accumulator store skip
                             "subs x24, x24, #0x8\n"
                             "add %x[res_ptr], %x[res_ptr], #0x20\n"
                             "bne 6b\n"
                             "subs x13, x13, #0x4\n"
                             "add %x[a_ptr], %x[a_ptr], x12\n"
                             "mov %x[res_ptr], x23\n"
                             "bgt 5b\n"
                             "9:" // Row tail: Row loop skip
                             : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
                             : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb),
                               [res_stride] "r"(res_stride), [nc] "r"(nc)
                             : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x12", "x13", "x20",
                               "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1",
                               "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12",
                               "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22",
                               "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
        return;
    }
    // else if (mllm_cpu_has_neon() && mllm_cpu_has_matmul_int8()) {
    //     assert((mllm_cpu_has_sve() && (svcntw() == 8))
    //            && "__ARM_FEATURE_SVE for vector size of 256-bits not defined, use the "
    //               "Q4_0_4_8 quantization format for optimal "
    //               "performance");
    // } else if (mllm_cpu_has_neon()) {
    //     assert(((mllm_cpu_has_sve() && (svcntw() == 8)) || mllm_cpu_has_matmul_int8())
    //            && "__ARM_FEATURE_SVE for vector size of 256-bits and "
    //               "__ARM_FEATURE_MATMUL_INT8 not defined, use the Q4_0_4_4 "
    //               "quantization format for optimal performance");
    // }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    assert(mllm_cpu_has_sve()
           && "__ARM_FEATURE_SVE not defined, use the Q4_0_4_8 quantization format "
              "for optimal performance");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    assert((mllm_cpu_has_sve() || mllm_cpu_has_matmul_int8())
           && "__ARM_FEATURE_SVE and __ARM_FEATURE_MATMUL_INT8 not defined, use the "
              "Q4_0_4_4 quantization format for optimal "
              "performance");
#else
    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        << 4);
                                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i])
                                         + (v1
                                            * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i
                                                          + qk / 2 * 4]))
                                        >> 4;
                            }
                            sumf[m][j] += sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j])
                                          * MLLM_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
#endif
}

void _gemm_q4_0_8x8_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc,
                              const void *__restrict bias) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    (void)s;
    (void)bs;
    (void)vx;
    (void)vy;
    (void)nr;
    (void)nc;
    (void)nb;
    (void)ncols_interleaved;
    (void)blocklen;

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    // if (svcntw() == 8) {
    if (true) {
        const void *b_ptr = vx;
        const void *a_ptr = vy;
        const void *bias_ptr = bias;
        float *res_ptr = s;
        size_t res_stride = bs * sizeof(float);

        __asm__ __volatile__("mov x20, #0x4\n"
                             "mov x13, %x[nr]\n"
                             "mov z28.s, #-0x4\n"
                             "mov x12, #0x88\n"
                             "ptrue p1.b\n"
                             "whilelt p0.s, XZR, x20\n"
                             "cmp x13, #0x10\n"
                             "mul x12, %x[nb], x12\n"
                             "blt 4f\n"
                             "1:" // Row loop
                             "add x11, %x[b_ptr], #0x10\n"
                             "mov x10, %x[nc]\n"
                             "add x9, %x[res_ptr], %x[res_stride], LSL #4\n"
                             "2:" // Column loop
                             "add x28, %x[a_ptr], #0x8\n"
                             "mov z24.b, #0x0\n"
                             "mov z15.b, #0x0\n"
                             "mov x27, %x[nb]\n"
                             "add x26, x28, x12\n"
                             "mov z12.b, #0x0\n"
                             "mov z0.b, #0x0\n"
                             "add x25, x26, x12\n"
                             "mov z13.b, #0x0\n"
                             "mov z1.b, #0x0\n"
                             "add x24, x25, x12\n"
                             "mov z20.b, #0x0\n"
                             "mov z25.b, #0x0\n"
                             "mov z11.b, #0x0\n"
                             "mov z16.b, #0x0\n"
                             "mov z19.b, #0x0\n"
                             "mov z26.b, #0x0\n"
                             "mov z8.b, #0x0\n"
                             "mov z29.b, #0x0\n"
                             "mov z27.b, #0x0\n"
                             "mov z10.b, #0x0\n"
                             "3:" // Block loop
                             "ld1b { z30.b }, p1/Z, [x11]\n"
                             "ld1b { z21.b }, p1/Z, [x11, #1, MUL VL]\n"
                             "mov z18.s, #0x0\n"
                             "mov z7.s, #0x0\n"
                             "ld1rqb { z3.b }, p1/Z, [x28]\n"
                             "ld1rqb { z5.b }, p1/Z, [x28, #16]\n"
                             "mov z9.s, #0x0\n"
                             "mov z22.s, #0x0\n"
                             "ld1b { z4.b }, p1/Z, [x11, #2, MUL VL]\n"
                             "ld1b { z17.b }, p1/Z, [x11, #3, MUL VL]\n"
                             "sub x20, x11, #0x10\n"
                             "sub x23, x28, #0x8\n"
                             "lsl z31.b, z30.b, #0x4\n"
                             "lsl z6.b, z21.b, #0x4\n"
                             "ld1h { z23.s }, p1/Z, [x20]\n"
                             "sub x22, x26, #0x8\n"
                             "and z30.b, z30.b, #0xf0\n"
                             "and z21.b, z21.b, #0xf0\n"
                             "sub x21, x25, #0x8\n"
                             "sub x20, x24, #0x8\n"
                             "lsl z14.b, z4.b, #0x4\n"
                             "lsl z2.b, z17.b, #0x4\n"
                             "subs x27, x27, #0x1\n"
                             "add x11, x11, #0x90\n"
                             ".inst 0x451f9872  // smmla z18.s, z3.b, z31.b\n"
                             ".inst 0x45069867  // smmla z7.s, z3.b, z6.b\n"
                             "ld1rqb { z3.b }, p1/Z, [x28, #32]\n"
                             "and z4.b, z4.b, #0xf0\n"
                             ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
                             ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x28, #48]\n"
                             "and z17.b, z17.b, #0xf0\n"
                             "fcvt z23.s, p1/m, z23.h\n"
                             ".inst 0x450e9872  // smmla z18.s, z3.b, z14.b\n"
                             ".inst 0x45029867  // smmla z7.s, z3.b, z2.b\n"
                             "ld1rqb { z3.b }, p1/Z, [x28, #64]\n"
                             ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
                             ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x28, #80]\n"
                             "fscale z23.s, p1/m, z23.s, z28.s\n"
                             ".inst 0x451e9872  // smmla z18.s, z3.b, z30.b\n"
                             ".inst 0x45159867  // smmla z7.s, z3.b, z21.b\n"
                             "ld1rqb { z3.b }, p1/Z, [x28, #96]\n"
                             ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
                             ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x28, #112]\n"
                             "add x28, x28, #0x88\n"
                             ".inst 0x45049872  // smmla z18.s, z3.b, z4.b\n"
                             ".inst 0x45119867  // smmla z7.s, z3.b, z17.b\n"
                             "ld1h { z3.s }, p0/Z, [x23]\n"
                             ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
                             ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
                             "fcvt z3.s, p1/m, z3.h\n"
                             "uzp1 z5.d, z18.d, z7.d\n"
                             "uzp2 z18.d, z18.d, z7.d\n"
                             "mov z3.q, z3.q[0]\n"
                             "uzp1 z7.d, z9.d, z22.d\n"
                             "uzp2 z22.d, z9.d, z22.d\n"
                             "fmul z9.s, z23.s, z3.s[0]\n"
                             "scvtf z5.s, p1/m, z5.s\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "scvtf z7.s, p1/m, z7.s\n"
                             "scvtf z22.s, p1/m, z22.s\n"
                             "fmla z24.s, p1/M, z5.s, z9.s\n"
                             "ld1rqb { z5.b }, p1/Z, [x26]\n"
                             "fmul z9.s, z23.s, z3.s[1]\n"
                             "fmla z15.s, p1/M, z18.s, z9.s\n"
                             "ld1rqb { z18.b }, p1/Z, [x26, #16]\n"
                             "fmul z9.s, z23.s, z3.s[2]\n"
                             "fmul z3.s, z23.s, z3.s[3]\n"
                             "fmla z12.s, p1/M, z7.s, z9.s\n"
                             "mov z9.s, #0x0\n"
                             "ld1h { z7.s }, p0/Z, [x22]\n"
                             ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
                             "fmla z0.s, p1/M, z22.s, z3.s\n"
                             "mov z22.s, #0x0\n"
                             "ld1h { z3.s }, p0/Z, [x21]\n"
                             ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x26, #32]\n"
                             "fcvt z7.s, p1/m, z7.h\n"
                             "fcvt z3.s, p1/m, z3.h\n"
                             ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
                             ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x26, #64]\n"
                             "mov z7.q, z7.q[0]\n"
                             "mov z3.q, z3.q[0]\n"
                             ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
                             ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x26, #96]\n"
                             ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
                             ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
                             "uzp1 z5.d, z9.d, z22.d\n"
                             "scvtf z5.s, p1/m, z5.s\n"
                             "uzp2 z22.d, z9.d, z22.d\n"
                             "fmul z9.s, z23.s, z7.s[0]\n"
                             "scvtf z22.s, p1/m, z22.s\n"
                             "fmla z13.s, p1/M, z5.s, z9.s\n"
                             "ld1rqb { z9.b }, p1/Z, [x25]\n"
                             "fmul z5.s, z23.s, z7.s[1]\n"
                             "fmla z1.s, p1/M, z22.s, z5.s\n"
                             "mov z5.s, #0x0\n"
                             "mov z22.s, #0x0\n"
                             ".inst 0x451f9a45  // smmla z5.s, z18.b, z31.b\n"
                             ".inst 0x45069a56  // smmla z22.s, z18.b, z6.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x26, #48]\n"
                             ".inst 0x450e9a45  // smmla z5.s, z18.b, z14.b\n"
                             ".inst 0x45029a56  // smmla z22.s, z18.b, z2.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x26, #80]\n"
                             ".inst 0x451e9a45  // smmla z5.s, z18.b, z30.b\n"
                             ".inst 0x45159a56  // smmla z22.s, z18.b, z21.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x26, #112]\n"
                             "add x26, x26, #0x88\n"
                             ".inst 0x45049a45  // smmla z5.s, z18.b, z4.b\n"
                             ".inst 0x45119a56  // smmla z22.s, z18.b, z17.b\n"
                             "uzp1 z18.d, z5.d, z22.d\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "uzp2 z22.d, z5.d, z22.d\n"
                             "fmul z5.s, z23.s, z7.s[2]\n"
                             "fmul z7.s, z23.s, z7.s[3]\n"
                             "scvtf z22.s, p1/m, z22.s\n"
                             "fmla z20.s, p1/M, z18.s, z5.s\n"
                             "ld1rqb { z18.b }, p1/Z, [x25, #16]\n"
                             "ld1h { z5.s }, p0/Z, [x20]\n"
                             "fcvt z5.s, p1/m, z5.h\n"
                             "fmla z25.s, p1/M, z22.s, z7.s\n"
                             "mov z22.s, #0x0\n"
                             "mov z7.s, #0x0\n"
                             ".inst 0x451f9936  // smmla z22.s, z9.b, z31.b\n"
                             ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
                             "ld1rqb { z9.b }, p1/Z, [x25, #32]\n"
                             "mov z5.q, z5.q[0]\n"
                             ".inst 0x450e9936  // smmla z22.s, z9.b, z14.b\n"
                             ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
                             "ld1rqb { z9.b }, p1/Z, [x25, #64]\n"
                             ".inst 0x451e9936  // smmla z22.s, z9.b, z30.b\n"
                             ".inst 0x45159927  // smmla z7.s, z9.b, z21.b\n"
                             "ld1rqb { z9.b }, p1/Z, [x25, #96]\n"
                             ".inst 0x45049936  // smmla z22.s, z9.b, z4.b\n"
                             ".inst 0x45119927  // smmla z7.s, z9.b, z17.b\n"
                             "uzp1 z9.d, z22.d, z7.d\n"
                             "scvtf z9.s, p1/m, z9.s\n"
                             "uzp2 z22.d, z22.d, z7.d\n"
                             "fmul z7.s, z23.s, z3.s[0]\n"
                             "scvtf z22.s, p1/m, z22.s\n"
                             "fmla z11.s, p1/M, z9.s, z7.s\n"
                             "ld1rqb { z9.b }, p1/Z, [x24]\n"
                             "fmul z7.s, z23.s, z3.s[1]\n"
                             "fmla z16.s, p1/M, z22.s, z7.s\n"
                             "mov z22.s, #0x0\n"
                             "mov z7.s, #0x0\n"
                             ".inst 0x451f9a56  // smmla z22.s, z18.b, z31.b\n"
                             ".inst 0x45069a47  // smmla z7.s, z18.b, z6.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x25, #48]\n"
                             ".inst 0x450e9a56  // smmla z22.s, z18.b, z14.b\n"
                             ".inst 0x45029a47  // smmla z7.s, z18.b, z2.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x25, #80]\n"
                             ".inst 0x451e9a56  // smmla z22.s, z18.b, z30.b\n"
                             ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x25, #112]\n"
                             "add x25, x25, #0x88\n"
                             ".inst 0x45049a56  // smmla z22.s, z18.b, z4.b\n"
                             ".inst 0x45119a47  // smmla z7.s, z18.b, z17.b\n"
                             "uzp1 z18.d, z22.d, z7.d\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "uzp2 z7.d, z22.d, z7.d\n"
                             "fmul z22.s, z23.s, z3.s[2]\n"
                             "fmul z3.s, z23.s, z3.s[3]\n"
                             "scvtf z7.s, p1/m, z7.s\n"
                             "fmla z19.s, p1/M, z18.s, z22.s\n"
                             "ld1rqb { z18.b }, p1/Z, [x24, #16]\n"
                             "fmul z22.s, z23.s, z5.s[0]\n"
                             "fmla z26.s, p1/M, z7.s, z3.s\n"
                             "mov z3.s, #0x0\n"
                             "mov z7.s, #0x0\n"
                             ".inst 0x451f9923  // smmla z3.s, z9.b, z31.b\n"
                             ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
                             "ld1rqb { z9.b }, p1/Z, [x24, #32]\n"
                             ".inst 0x450e9923  // smmla z3.s, z9.b, z14.b\n"
                             ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
                             "mov z9.s, #0x0\n"
                             ".inst 0x451f9a49  // smmla z9.s, z18.b, z31.b\n"
                             "mov z31.s, #0x0\n"
                             ".inst 0x45069a5f  // smmla z31.s, z18.b, z6.b\n"
                             "ld1rqb { z6.b }, p1/Z, [x24, #48]\n"
                             "ld1rqb { z18.b }, p1/Z, [x24, #64]\n"
                             ".inst 0x450e98c9  // smmla z9.s, z6.b, z14.b\n"
                             "fmul z14.s, z23.s, z5.s[1]\n"
                             ".inst 0x450298df  // smmla z31.s, z6.b, z2.b\n"
                             "ld1rqb { z6.b }, p1/Z, [x24, #80]\n"
                             "fmul z2.s, z23.s, z5.s[2]\n"
                             "fmul z23.s, z23.s, z5.s[3]\n"
                             ".inst 0x451e9a43  // smmla z3.s, z18.b, z30.b\n"
                             ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
                             "ld1rqb { z5.b }, p1/Z, [x24, #96]\n"
                             ".inst 0x451e98c9  // smmla z9.s, z6.b, z30.b\n"
                             ".inst 0x451598df  // smmla z31.s, z6.b, z21.b\n"
                             "ld1rqb { z18.b }, p1/Z, [x24, #112]\n"
                             "add x24, x24, #0x88\n"
                             ".inst 0x450498a3  // smmla z3.s, z5.b, z4.b\n"
                             ".inst 0x451198a7  // smmla z7.s, z5.b, z17.b\n"
                             ".inst 0x45049a49  // smmla z9.s, z18.b, z4.b\n"
                             ".inst 0x45119a5f  // smmla z31.s, z18.b, z17.b\n"
                             "uzp1 z18.d, z3.d, z7.d\n"
                             "uzp2 z5.d, z3.d, z7.d\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "uzp1 z6.d, z9.d, z31.d\n"
                             "uzp2 z9.d, z9.d, z31.d\n"
                             "scvtf z5.s, p1/m, z5.s\n"
                             "fmla z8.s, p1/M, z18.s, z22.s\n"
                             "scvtf z6.s, p1/m, z6.s\n"
                             "scvtf z9.s, p1/m, z9.s\n"
                             "fmla z29.s, p1/M, z5.s, z14.s\n"
                             "fmla z27.s, p1/M, z6.s, z2.s\n"
                             "fmla z10.s, p1/M, z9.s, z23.s\n"
                             "bgt 3b\n"
                             "mov x20, %x[res_ptr]\n"
                             "subs x10, x10, #0x8\n"
                             "add %x[res_ptr], %x[res_ptr], #0x20\n"
                             "st1w { z24.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z15.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z12.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z0.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z13.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z1.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z20.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z25.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z11.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z16.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z19.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z26.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z8.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z29.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z27.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "st1w { z10.s }, p1, [x20]\n"
                             "bne 2b\n"
                             "mov x20, #0x4\n"
                             "sub x13, x13, #0x10\n"
                             "cmp x13, #0x10\n"
                             "mov %x[res_ptr], x9\n"
                             "madd %x[a_ptr], x20, x12, %x[a_ptr]\n"
                             "bge 1b\n"
                             "4:" // Row loop skip
                             "cbz x13, 9f\n"
                             "5:" // Row tail: Row loop
                             "add x25, %x[b_ptr], #0x10\n"
                             "mov x24, %x[nc]\n"
                             "add x23, %x[res_ptr], %x[res_stride], LSL #2\n"
                             "6:" // Row tail: Column loop
                             "mov z24.b, #0x0\n"
                             "mov z15.b, #0x0\n"
                             "add x28, %x[a_ptr], #0x8\n"
                             "mov x22, %x[nb]\n"
                             "mov z12.b, #0x0\n"
                             "mov z0.b, #0x0\n"
                             "7:" // Row tail: Block loop
                             "ld1b { z3.b }, p1/Z, [x25]\n"
                             "ld1b { z6.b }, p1/Z, [x25, #1, MUL VL]\n"
                             "mov z2.s, #0x0\n"
                             "mov z25.s, #0x0\n"
                             "ld1rqb { z26.b }, p1/Z, [x28]\n"
                             "ld1rqb { z21.b }, p1/Z, [x28, #16]\n"
                             "mov z27.s, #0x0\n"
                             "mov z19.s, #0x0\n"
                             "ld1b { z29.b }, p1/Z, [x25, #2, MUL VL]\n"
                             "ld1b { z16.b }, p1/Z, [x25, #3, MUL VL]\n"
                             "sub x21, x25, #0x10\n"
                             "sub x20, x28, #0x8\n"
                             "lsl z20.b, z3.b, #0x4\n"
                             "lsl z4.b, z6.b, #0x4\n"
                             "ld1rqb { z10.b }, p1/Z, [x28, #32]\n"
                             "ld1rqb { z23.b }, p1/Z, [x28, #48]\n"
                             "and z3.b, z3.b, #0xf0\n"
                             "and z6.b, z6.b, #0xf0\n"
                             "ld1rqb { z11.b }, p1/Z, [x28, #64]\n"
                             "ld1rqb { z7.b }, p1/Z, [x28, #80]\n"
                             "lsl z8.b, z29.b, #0x4\n"
                             "lsl z14.b, z16.b, #0x4\n"
                             "ld1rqb { z18.b }, p1/Z, [x28, #96]\n"
                             "ld1rqb { z30.b }, p1/Z, [x28, #112]\n"
                             ".inst 0x45149b42  // smmla z2.s, z26.b, z20.b\n"
                             ".inst 0x45049b59  // smmla z25.s, z26.b, z4.b\n"
                             "and z29.b, z29.b, #0xf0\n"
                             "ld1h { z17.s }, p1/Z, [x21]\n"
                             ".inst 0x45149abb  // smmla z27.s, z21.b, z20.b\n"
                             ".inst 0x45049ab3  // smmla z19.s, z21.b, z4.b\n"
                             "and z16.b, z16.b, #0xf0\n"
                             "ld1h { z4.s }, p0/Z, [x20]\n"
                             "subs x22, x22, #0x1\n"
                             "add x28, x28, #0x88\n"
                             "fcvt z17.s, p1/m, z17.h\n"
                             "add x25, x25, #0x90\n"
                             ".inst 0x45089942  // smmla z2.s, z10.b, z8.b\n"
                             ".inst 0x450e9959  // smmla z25.s, z10.b, z14.b\n"
                             "fcvt z4.s, p1/m, z4.h\n"
                             ".inst 0x45089afb  // smmla z27.s, z23.b, z8.b\n"
                             ".inst 0x450e9af3  // smmla z19.s, z23.b, z14.b\n"
                             "fscale z17.s, p1/m, z17.s, z28.s\n"
                             "mov z4.q, z4.q[0]\n"
                             ".inst 0x45039962  // smmla z2.s, z11.b, z3.b\n"
                             ".inst 0x45069979  // smmla z25.s, z11.b, z6.b\n"
                             "fmul z23.s, z17.s, z4.s[0]\n"
                             "fmul z9.s, z17.s, z4.s[1]\n"
                             "fmul z21.s, z17.s, z4.s[2]\n"
                             "fmul z4.s, z17.s, z4.s[3]\n"
                             ".inst 0x450398fb  // smmla z27.s, z7.b, z3.b\n"
                             ".inst 0x450698f3  // smmla z19.s, z7.b, z6.b\n"
                             ".inst 0x451d9a42  // smmla z2.s, z18.b, z29.b\n"
                             ".inst 0x45109a59  // smmla z25.s, z18.b, z16.b\n"
                             ".inst 0x451d9bdb  // smmla z27.s, z30.b, z29.b\n"
                             ".inst 0x45109bd3  // smmla z19.s, z30.b, z16.b\n"
                             "uzp1 z31.d, z2.d, z25.d\n"
                             "uzp2 z13.d, z2.d, z25.d\n"
                             "scvtf z31.s, p1/m, z31.s\n"
                             "uzp1 z17.d, z27.d, z19.d\n"
                             "uzp2 z18.d, z27.d, z19.d\n"
                             "scvtf z13.s, p1/m, z13.s\n"
                             "fmla z24.s, p1/M, z31.s, z23.s\n"
                             "scvtf z17.s, p1/m, z17.s\n"
                             "scvtf z18.s, p1/m, z18.s\n"
                             "fmla z15.s, p1/M, z13.s, z9.s\n"
                             "fmla z12.s, p1/M, z17.s, z21.s\n"
                             "fmla z0.s, p1/M, z18.s, z4.s\n"
                             "bgt 7b\n"
                             "mov x20, %x[res_ptr]\n"
                             "cmp x13, #0x1\n"
                             "st1w { z24.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "ble 8f\n"
                             "cmp x13, #0x2\n"
                             "st1w { z15.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "ble 8f\n"
                             "cmp x13, #0x3\n"
                             "st1w { z12.s }, p1, [x20]\n"
                             "add x20, x20, %x[res_stride]\n"
                             "ble 8f\n"
                             "st1w { z0.s }, p1, [x20]\n"
                             "8:" // Row tail: Accumulator store skip
                             "subs x24, x24, #0x8\n"
                             "add %x[res_ptr], %x[res_ptr], #0x20\n"
                             "bne 6b\n"
                             "subs x13, x13, #0x4\n"
                             "add %x[a_ptr], %x[a_ptr], x12\n"
                             "mov %x[res_ptr], x23\n"
                             "bgt 5b\n"
                             "9:" // Row tail: Row loop skip
                             : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
                             : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb),
                               [res_stride] "r"(res_stride), [nc] "r"(nc)
                             : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x12", "x13", "x20",
                               "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1",
                               "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12",
                               "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22",
                               "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
        return;
    }
    // else if (mllm_cpu_has_neon() && mllm_cpu_has_matmul_int8()) {
    //     assert((mllm_cpu_has_sve() && (svcntw() == 8))
    //            && "__ARM_FEATURE_SVE for vector size of 256-bits not defined, use the "
    //               "Q4_0_4_8 quantization format for optimal "
    //               "performance");
    // } else if (mllm_cpu_has_neon()) {
    //     assert(((mllm_cpu_has_sve() && (svcntw() == 8)) || mllm_cpu_has_matmul_int8())
    //            && "__ARM_FEATURE_SVE for vector size of 256-bits and "
    //               "__ARM_FEATURE_MATMUL_INT8 not defined, use the Q4_0_4_4 "
    //               "quantization format for optimal performance");
    // }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    assert(mllm_cpu_has_sve()
           && "__ARM_FEATURE_SVE not defined, use the Q4_0_4_8 quantization format "
              "for optimal performance");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    assert((mllm_cpu_has_sve() || mllm_cpu_has_matmul_int8())
           && "__ARM_FEATURE_SVE and __ARM_FEATURE_MATMUL_INT8 not defined, use the "
              "Q4_0_4_4 quantization format for optimal "
              "performance");
#else
    float sumf[4][8];
    int sumi;

    const float *bias_ptr = (const float *)bias;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    sumf[m][j] = bias_ptr[x * ncols_interleaved + j];
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        << 4);
                                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen
                                                                    + j * blocklen + i]
                                                        & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i])
                                         + (v1
                                            * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i
                                                          + qk / 2 * 4]))
                                        >> 4;
                            }
                            sumf[m][j] += sumi * MLLM_FP16_TO_FP32(b_ptr[l].d[j])
                                          * MLLM_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
#endif
}

void quantize_row_q4_0_4x4(const float *__restrict x, void *__restrict y, int k) {
    assert(k % QK4_0 == 0);
    std::cout << "Quantize 4x4:" << k << "/4096=" << k / 4096;
    auto size = quantize_q4_0_nr_bl(x, y, k / 4096, 4096, 4, 4);
}

void quantize_row_q4_0_4x4(const float *__restrict x, void *__restrict y, int k, int raw) {
    assert(k % QK4_0 == 0);
    std::cout << "Quantize 4x4:" << k << "/" << raw << "=" << k / raw;
    auto size = quantize_q4_0_nr_bl(x, y, k / raw, raw, 4, 4);
}