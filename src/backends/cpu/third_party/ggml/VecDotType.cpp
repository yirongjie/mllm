/*
 * This code is based on ggml(https://github.com/ggerganov/ggml),
 * please see https://github.com/ggerganov/ggml/blob/master/src/ggml.c
 * ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
 *
 * MIT License
 * Copyright (c) 2022 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include "VecDotType.hpp"
#include "Types.hpp"
#include "Quantize.hpp"
#include "VecDotFP32.hpp"
#include "VecDotFP16.hpp"
#include "VecDotQ2.hpp"
#include "VecDotQ3.hpp"
#include "VecDotQ4.hpp"
#include "VecDotQ6.hpp"
#include "VecDotQ8.hpp"
#include "GemmPack.hpp"

void fp32_add_row_to(int n, const float *MLLM_RESTRICT src, float *MLLM_RESTRICT dst, float alpha) {
    int i = 0;
#ifdef __AVX2__
    __m256 alpha_vec = _mm256_set1_ps(alpha); // load alpha into 8 float register

    // 主循环处理8的倍数个元素
    for (; i <= n - 8; i += 8) {
        __m256 src_vec = _mm256_loadu_ps(src + i);                     // load 8 float from src
        __m256 dst_vec = _mm256_loadu_ps(dst + i);                     // load 8 float from dst
        __m256 res_vec = _mm256_fmadd_ps(src_vec, alpha_vec, dst_vec); // alpha * src + dst
        _mm256_storeu_ps(dst + i, res_vec);                            // store back to dst
    }
#elif defined(__ARM_NEON)
    // TODO: generated by GPT-4, not tested yet
    float32x4_t alpha_vec = vdupq_n_f32(alpha); // load alpha into all elements of a 128-bit register

    // Main loop for multiples of 4
    for (; i <= n - 4; i += 4) {
        float32x4_t src_vec = vld1q_f32(src + i);
        float32x4_t dst_vec = vld1q_f32(dst + i);
        float32x4_t res_vec = vmlaq_f32(dst_vec, src_vec, alpha_vec); // calculate alpha * src + dst
        vst1q_f32(dst + i, res_vec);                                  // store result back to dst
    }
#endif

    // 处理剩余的元素
    for (; i < n; ++i) {
        dst[i] = dst[i] + alpha * src[i];
    }
}

void fp_16_add_row_to(int n, const mllm_fp16_t *MLLM_RESTRICT src, float *MLLM_RESTRICT dst, float alpha) {
    int i = 0;
#ifdef __AVX2__
    __m256 alpha_vec = _mm256_set1_ps(alpha); // load alpha into 8 float register

    // 主循环处理8的倍数个元素
    for (; i <= n - 8; i += 8) {
        __m128i src_fp16 = _mm_loadu_si128((__m128i const *)(src + i)); // load 8 fp16 from src
        __m256 src_vec = _mm256_cvtph_ps(src_fp16);                     // convert to 8 fp32
        __m256 dst_vec = _mm256_loadu_ps(dst + i);                      // load 8 float from dst
        __m256 res_vec = _mm256_fmadd_ps(src_vec, alpha_vec, dst_vec);  // alpha * src + dst
        _mm256_storeu_ps(dst + i, res_vec);                             // store back to dst
    }
#elif defined(__ARM_NEON)
    std::cout << "not support now" << std::endl;
#endif

    // 处理剩余的元素
    for (; i < n; ++i) {
        dst[i] = dst[i] + alpha * MLLM_FP16_TO_FP32(src[i]);
    }
}

void q4_0_add_row_to(int n, const block_q4_0 *MLLM_RESTRICT src, float *MLLM_RESTRICT dst, float alpha) {
    assert(n % QK4_0 == 0);
    auto num_blocks = n / QK4_0;

    int i = 0;
#ifdef __AVX2__
    // TODO: not implemented
#elif defined(__ARM_NEON)
    // TODO: not implemented
#endif

    // Process the remaining elements
    for (; i < num_blocks; ++i) {
        auto scale = MLLM_FP16_TO_FP32(src[i].d) * alpha;
        auto offset = i * QK4_0;

        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int v0 = (src[i].qs[j] & 0x0F) - 8;
            const int v1 = (src[i].qs[j] >> 4) - 8;
            dst[offset + j] = dst[offset + j] + (scale * static_cast<float>(v0));
            dst[offset + j + QK4_0 / 2] = dst[offset + j + QK4_0 / 2] + (scale * static_cast<float>(v1));
        }
    }
}

#if QK_K == 256
static inline void get_scale_min_k4(int j, const uint8_t *__restrict q, uint8_t *__restrict d, uint8_t *__restrict m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
#endif

void q4_k_add_row_to(int n, const block_q4_K *MLLM_RESTRICT src, float *MLLM_RESTRICT dst, float alpha) {
#if QK_K == 256
    assert(n % QK_K == 0);
    assert(QK_K == 256); // TODO: It is wired here for now
    const int nb = n / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q = src[i].qs;

        const float d = MLLM_FP16_TO_FP32(src[i].d);      // scale for super block's d
        const float min = MLLM_FP16_TO_FP32(src[i].dmin); // scale for super block's min

        int is = 0;
        uint8_t sc;
        uint8_t m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, src[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, src[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                *dst = *dst + (d1 * (q[l] & 0xF) - m1) * alpha;
                dst++;
            }
            for (int l = 0; l < 32; ++l) {
                *dst = *dst + (d2 * (q[l] >> 4) - m2) * alpha;
                dst++;
            }
            q += 32;
            is += 2;
        }
    }
#endif
}

void q6_k_add_row_to(int n, const block_q6_K *MLLM_RESTRICT src, float *MLLM_RESTRICT dst, float alpha) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = MLLM_FP16_TO_FP32(src[i].d);
        const float scale = d * alpha;

        const uint8_t *__restrict ql = src[i].ql;
        const uint8_t *__restrict qh = src[i].qh;
        const int8_t *__restrict sc = src[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[l + 0] += scale * sc[is + 0] * q1;
                dst[l + 32] += scale * sc[is + 2] * q2;
                dst[l + 64] += scale * sc[is + 4] * q3;
                dst[l + 96] += scale * sc[is + 6] * q4;
            }
            dst += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void q8_0_add_row_to(int n, const block_q8_0 *MLLM_RESTRICT src, float *MLLM_RESTRICT dst, float alpha) {
    static const int qk = QK8_0;

    assert(n % qk == 0);

    const int nb = n / qk;

    const block_q8_0 *__restrict x = src;

    for (int i = 0; i < nb; i++) {
        const float scale = MLLM_FP16_TO_FP32(x[i].d) * alpha;

        for (int j = 0; j < qk; ++j) {
            dst[i * qk + j] += x[i].qs[j] * scale;
        }
    }
}

void q8_k_add_row_to(int n, const block_q8_K *MLLM_RESTRICT src, float *MLLM_RESTRICT dst, float alpha) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    for (int i = 0; i < nb; i++) {
        auto scale = src[i].d * alpha;
        for (int j = 0; j < QK_K; ++j) {
            *dst++ += scale * src[i].qs[j];
        }
    }
}

/***
 * This is the type traits for different data types used in MLLM.
 * It defines the size, block size, conversion functions, vector dot product functions,
 * and row addition functions for each data type.
 *
 * The `type_traits` array is indexed by the `mllm_type` enum values.
 * Each entry in the array corresponds to a specific data type and contains
 * the necessary information to handle that type.  * The `to_float` and `from_float` functions are used to convert between
 * the data type and float representation.
 * The `vec_dot` function is used to compute the dot product of two vectors of the
 * specified data type.
 * The `add_row_to` function is used to add a row of the specified data type to a destination vector.
 * The `vec_dot_type` field specifies the data type used for the dot product.
 * The `size` field specifies the size of the data type in bytes.
 * The `blck_size` field specifies the block size of the data type.
 */

type_traits_t type_traits[] = {
    /*[MLLM_TYPE_F32] = */ {
        .size = sizeof(float),
        .blck_size = 1,
        .to_float = nullptr,
        .from_float = nullptr,
        .vec_dot = (mllm_vec_dot_func)vec_dot_fp32,
        .vec_dot_type = MLLM_TYPE_F32,
        .add_row_to = (mllm_vec_add_row_func)fp32_add_row_to,
    },
    /*[MLLM_TYPE_F16] = */ {
        .size = sizeof(mllm_fp16_t),
        .blck_size = 1,
        .to_float = (mllm_to_float_func)mllm_fp16_to_fp32_row,
        .from_float = (mllm_from_float_func)mllm_fp32_to_fp16_row,
        .vec_dot = (mllm_vec_dot_func)vec_dot_fp16,
        .vec_dot_type = MLLM_TYPE_F16,
        .add_row_to = (mllm_vec_add_row_func)fp_16_add_row_to,
    },
    /*[MLLM_TYPE_Q4_0] = */ {
        .size = sizeof(block_q4_0),
        .blck_size = QK4_0,
        .to_float = (mllm_to_float_func)dequantize_row_q4_0,
        .from_float = (mllm_from_float_func)quantize_row_q4_0,
        .vec_dot = (mllm_vec_dot_func)vec_dot_q4_0_q8_0,
        .vec_dot_type = MLLM_TYPE_Q8_0,
        .add_row_to = (mllm_vec_add_row_func)q4_0_add_row_to,
    },
    /*[MLLM_TYPE_Q4_1] = */ {
        // TODO: not implemented. It seems that it is not used in the current code
    },
    {},
    {},
    {},
    {},
    /*[MLLM_TYPE_Q8_0] = */ {
        .size = sizeof(block_q8_0),
        .blck_size = QK8_0,
        .to_float = (mllm_to_float_func)dequantize_row_q8_0,
        .from_float = (mllm_from_float_func)quantize_row_q8_0,
        .from_float_to_mat = (mllm_from_float_to_mat_func)quantize_mat_q8_0,
        .vec_dot = (mllm_vec_dot_func)vec_dot_q8_0_q8_0,
        .vec_dot_type = MLLM_TYPE_Q8_0,
        .add_row_to = (mllm_vec_add_row_func)q8_0_add_row_to,
    },
    /*[MLLM_TYPE_Q8_1] = */ {},
    {},
    {},
    /*[MLLM_TYPE_Q4_K] = */ {
        .size = sizeof(block_q4_K),
        .blck_size = QK_K,
        .to_float = (mllm_to_float_func)dequantize_row_q4_K,
        .from_float = (mllm_from_float_func)quantize_row_q4_K,
        .vec_dot = (mllm_vec_dot_func)vec_dot_q4_K_q8_K,
        .vec_dot_type = MLLM_TYPE_Q8_K,
        .add_row_to = (mllm_vec_add_row_func)q4_k_add_row_to,
    },
    {},
    /*[MLLM_TYPE_Q6_K] = */ {
        .size = sizeof(block_q6_K),
        .blck_size = QK_K,
        .to_float = (mllm_to_float_func)dequantize_row_q6_K,
        .from_float = (mllm_from_float_func)quantize_row_q6_K,
        .vec_dot = (mllm_vec_dot_func)vec_dot_q6_K_q8_K,
        .vec_dot_type = MLLM_TYPE_Q8_K,
        .add_row_to = (mllm_vec_add_row_func)q6_k_add_row_to,
    },
    /*[MLLM_TYPE_Q8_K] = */ {
        .size = sizeof(block_q8_K),
        .blck_size = QK_K,
        .to_float = (mllm_to_float_func)dequantize_row_q8_K,
        .from_float = (mllm_from_float_func)quantize_row_q8_K,
        .vec_dot = (mllm_vec_dot_func) nullptr, // TODO: not implemented, no need to implement now
        .vec_dot_type = MLLM_TYPE_Q8_K,
        .add_row_to = (mllm_vec_add_row_func)q8_k_add_row_to,
    },
    /*[MLLM_TYPE_I_8] = */ {}, // TODO: not implemented, integrate linear-int8 to common linear
    /*[MLLM_TYPE_I_16] = */ {},
    /*[MLLM_TYPE_I_32] = */ {},
    /*[MLLM_TYPE_Q4_0_4_4] = */ {
        .size = sizeof(block_q4_0),
        .blck_size = QK4_0,
        .blck_size_interleave = 4,
        .to_float = NULL,
        .from_float = NULL,
        .vec_dot = NULL,
        .vec_dot_type = MLLM_TYPE_Q8_0,
        // .nrows                    = 1,
        // .ncols                    = 4,
        .gemv = (gemv_func)gemv_q4_0_4x4_q8_0,
        .gemm = (gemm_func)gemm_q4_0_4x4_q8_0,
    },
    /*[MLLM_TYPE_Q4_0_4_8] = */ {
        .size = sizeof(block_q4_0),
        .blck_size = QK4_0,
        .blck_size_interleave = 8,
        // .is_quantized             = true,
        .to_float = NULL,
        .from_float = NULL,
        .vec_dot = NULL,
        .vec_dot_type = MLLM_TYPE_Q8_0,
        // .nrows                    = 1,
        // .ncols                    = 4,
        .gemv = (gemv_func)gemv_q4_0_4x8_q8_0,
        .gemm = (gemm_func)gemm_q4_0_4x8_q8_0,
    },
    /*[MLLM_TYPE_Q4_0_8_8] = */ {
        .size = sizeof(block_q4_0),
        .blck_size = QK4_0,
        .blck_size_interleave = 8,
        // .is_quantized             = true,
        .to_float = NULL,
        .from_float = NULL,
        .vec_dot = NULL,
        .vec_dot_type = MLLM_TYPE_Q8_0,
        // .nrows                    = 1,
        // .ncols                    = 8,
        .gemv = (gemv_func)gemv_q4_0_8x8_q8_0,
        .gemm = (gemm_func)gemm_q4_0_8x8_q8_0,
    },
    {},
    /*MLLM_TYPE_Q3_K = */ {
        .size = sizeof(block_q3_K),
        .blck_size = QK_K,
        .to_float = (mllm_to_float_func)dequantize_row_q3_K,
        .from_float = (mllm_from_float_func)quantize_row_q3_K,
        .vec_dot = (mllm_vec_dot_func)vec_dot_q3_K_q8_K,
        .vec_dot_type = MLLM_TYPE_Q8_K,
        .add_row_to = NULL, //(mllm_vec_add_row_func)q3_k_add_row_to,
    },
    /*MLLM_TYPE_Q2_K = */ {
        .size = sizeof(block_q2_K),
        .blck_size = QK_K,
        .to_float = (mllm_to_float_func)dequantize_row_q2_K,
        .from_float = (mllm_from_float_func)quantize_row_q2_K,
        .vec_dot = (mllm_vec_dot_func)vec_dot_q2_K_q8_K,
        .vec_dot_type = MLLM_TYPE_Q8_K,
        .add_row_to = NULL, //(mllm_vec_add_row_func)q2_k_add_row_to,

    },
    /*MLLM_TYPE_Q1_K = */ {},
    /*MLLM_TYPE_IQ2_XXS = */ {
        .size = sizeof(block_iq2_xxs),
        .blck_size = QK_K,
        .to_float = (mllm_to_float_func)dequantize_row_iq2_xxs,
        .from_float = NULL, //(mllm_from_float_func)quantize_row_iq2_xxs,
        .vec_dot = (mllm_vec_dot_func)vec_dot_iq2_xxs_q8_K,
        .vec_dot_type = MLLM_TYPE_Q8_K,
        .add_row_to = NULL, //(mllm_vec_add_row_func)q6_k_add_row_to,

    },
    {},
    {},
    {},
    /*MLLM_TYPE_IQ2_S = */ {},
    /*MLLM_TYPE_KLEIDIAI_Q4_0 = */ {},
    /*MLLM_TYPE_Q8_0F = */ {},
    /*MLLM_TYPE_Q2_0 = */ {
        .size = sizeof(block_q2_0),
        .blck_size = QK_K,
        .to_float = (mllm_to_float_func)dequantize_row_q2_0,
        .from_float = (mllm_from_float_func)quantize_row_q2_0,
        .vec_dot = (mllm_vec_dot_func)vec_dot_q2_0_q8_0,
        .vec_dot_type = MLLM_TYPE_Q8_0,
        .add_row_to = NULL, //(mllm_vec_add_row_func)q2_k_add_row_to,
    },
    // TODO: add support to more type
};
