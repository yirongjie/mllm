

// 文件名: kernel/div.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// ==================================================================
// 1. Tensor / Tensor Division Kernels (FP32 - 保持不变)
// ==================================================================

__kernel void div_float(
    __global const float *A,
    __global const float *B,
    __global float *C) {
    size_t index = get_global_id(0);
    // 添加保护防止除以零
    C[index] = B[index] == 0.0f ? 0.0f : A[index] / B[index];
}

__kernel void div_float_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    float4 inA = read_imagef(inputA, sampler, pos);
    float4 inB = read_imagef(inputB, sampler, pos);
    // 添加保护防止除以零
    float4 result = (inB.x == 0.0f && inB.y == 0.0f && inB.z == 0.0f && inB.w == 0.0f) ?
                        (float4)(0.0f) :
                        inA / inB;
    write_imagef(output, pos, result);
}

// ==================================================================
// 2. Tensor / Scalar Division Kernels (FP32 - 保持不变)
// ==================================================================

__kernel void div_scalar_float(
    __global const float *A,
    const float B,
    __global float *C) {
    size_t index = get_global_id(0);
    C[index] = B == 0.0f ? 0.0f : A[index] / B;
}

__kernel void div_scalar_float_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const float B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    float4 inA = read_imagef(inputA, sampler, pos);
    float4 inB = (float4)(B);
    float4 result = (B == 0.0f) ? (float4)(0.0f) : inA / inB;
    write_imagef(output, pos, result);
}

// ==================================================================
// FP16 Kernels Implementations with Preprocessor Guards
// ==================================================================

#ifdef SUPPORTS_FP16

__kernel void div_fp16_vector(
    __global const half *A,
    __global const half *B,
    __global half *C) {
    const int i = get_global_id(0);
    half4 a_vec = vload4(i, A);
    half4 b_vec = vload4(i, B);
    // 添加保护防止除以零 (转换为 float 进行比较)
    half4 c_vec = (all(convert_float4(b_vec) == 0.0f)) ? (half4)(0.0h) : a_vec / b_vec;
    vstore4(c_vec, i, C);
}

__kernel void div_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    half4 inA = read_imageh(inputA, sampler, pos);
    half4 inB = read_imageh(inputB, sampler, pos);
    half4 result = (all(convert_float4(inB) == 0.0f)) ? (half4)(0.0h) : inA / inB;
    write_imageh(output, pos, result);
}

__kernel void div_scalar_fp16_vector(
    __global const half *A,
    const float B, // <--- 修改点
    __global half *C) {
    const int i = get_global_id(0);
    float4 a_vec_f = convert_float4(vload4(i, A));

    // B 已经是 float，无需转换
    float4 c_vec_f = a_vec_f / B;

    vstore4(convert_half4_rte(c_vec_f), i, C);
}

__kernel void div_scalar_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const float B, // <--- 修改点
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }

    // 1. 读取 half4 数据并立即提升到 float4
    float4 inA_f = convert_float4(read_imageh(inputA, sampler, pos));

    // 2. 在 float 精度下进行计算
    float4 result_f = inA_f / B;

    // 3. 将结果转换回 half4 并写入
    write_imageh(output, pos, convert_half4_rte(result_f));
}

#else // !SUPPORTS_FP16

// ===================== B. FP16实现 (软件回退) =====================
// 当硬件不支持时, 不使用'half'类型.
// 我们用'ushort'来存储16位数据, 并手动转换到'float'进行计算.

// 帮助函数: 将 ushort (存储的half) 转换为 float
// inline float half_to_float(ushort h) {
//     uint s = (h & 0x8000) << 16;
//     uint e = (h & 0x7C00) >> 10;
//     uint f = h & 0x03FF;

//     if (e == 0) {
//         if (f == 0) return as_float(s);
//         e = 113;
//         while ((f & 0x0400) == 0) {
//             f <<= 1;
//             e--;
//         }
//         f &= 0x03FF;
//     } else if (e == 0x1F) {
//         return as_float(s | 0x7F800000 | (f << 13));
//     } else {
//         e += 112;
//     }
//     return as_float(s | (e << 23) | (f << 13));
// }

inline float half_to_float(ushort h) {
    const uint s = (h >> 15) & 0x0001;
    const uint e = (h >> 10) & 0x001f;
    const uint f = h & 0x03ff;
    uint float_val;

    if (e == 0) {
        if (f == 0) { // +0 or -0
            float_val = s << 31;
        } else { // Denormalized number to normalized float
            uint f_shifted = f;
            uint e_shifted = e;
            while ((f_shifted & 0x0400) == 0) {
                f_shifted <<= 1;
                e_shifted--;
            }
            // 此时 e_shifted 是有效指数的偏移量, f_shifted 的第10位是1
            e_shifted++;          // 补偿
            f_shifted &= ~0x0400; // 移除前导的1
            // 加上新的指数偏置 (127 - 15), 再加上有效指数偏移
            float_val = (s << 31) | ((e_shifted + 112) << 23) | (f_shifted << 13);
        }
    } else if (e == 31) { // Inf or NaN
        if (f == 0) {     // +/- Infinity
            float_val = (s << 31) | 0x7f800000;
        } else { // NaN
            float_val = (s << 31) | 0x7f800000 | (f << 13);
        }
    } else { // Normalized number
        float_val = (s << 31) | ((e + 112) << 23) | (f << 13);
    }

    return as_float(float_val);
}

// 帮助函数: 将 float 转换为 ushort (存储为half)
inline ushort float_to_half(float f) {
    uint u = as_uint(f);
    uint s = (u >> 16) & 0x8000;
    int e = ((u >> 23) & 0xFF) - 127;
    uint f_mant = u & 0x7FFFFF;

    if (e > 15) return (ushort)(s | 0x7C00);
    if (e < -14) {
        f_mant |= 0x800000;
        return (ushort)(s | (f_mant >> (-e - 14)));
    }
    return (ushort)(s | ((e + 15) << 10) | (f_mant >> 13));
}

__kernel void div_fp16_vector(
    __global const ushort *A,
    __global const ushort *B,
    __global ushort *C) {
    const int i = get_global_id(0) * 4;

    float4 a_vec = (float4)(half_to_float(A[i]), half_to_float(A[i + 1]), half_to_float(A[i + 2]), half_to_float(A[i + 3]));
    float4 b_vec = (float4)(half_to_float(B[i]), half_to_float(B[i + 1]), half_to_float(B[i + 2]), half_to_float(B[i + 3]));

    float4 c_vec;
    c_vec.x = b_vec.x == 0.0f ? 0.0f : a_vec.x / b_vec.x;
    c_vec.y = b_vec.y == 0.0f ? 0.0f : a_vec.y / b_vec.y;
    c_vec.z = b_vec.z == 0.0f ? 0.0f : a_vec.z / b_vec.z;
    c_vec.w = b_vec.w == 0.0f ? 0.0f : a_vec.w / b_vec.w;

    C[i] = float_to_half(c_vec.x);
    C[i + 1] = float_to_half(c_vec.y);
    C[i + 2] = float_to_half(c_vec.z);
    C[i + 3] = float_to_half(c_vec.w);
}

__kernel void div_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    __read_only image2d_t inputB,
    __write_only image2d_t output,
    const int width,
    const int height) {
    // 这是一个存根(stub)实现, 因为不支持cl_khr_fp16的平台
    // 通常也不支持CL_HALF_FLOAT图像格式. 主机代码中该路径已通过&& false禁用.
    // 仅用于保证内核能被创建.
    return;
}

__kernel void div_scalar_fp16_vector(
    __global const half *A,
    const float B,
    __global half *C) {
    // 每个工作项依然负责4个元素，但我们将逐个处理它们
    const int i = get_global_id(0) * 4;

    // 临时存储4个float类型的结果
    float results[4];

    // 核心安全检查
    if (B == 0.0f) {
        results[0] = 0.0f;
        results[1] = 0.0f;
        results[2] = 0.0f;
        results[3] = 0.0f;
    } else {
        // 【关键改动】像 flash_attention.cl 一样，逐个加载、转换、计算
        results[0] = (float)A[i + 0] / B;
        results[1] = (float)A[i + 1] / B;
        results[2] = (float)A[i + 2] / B;
        results[3] = (float)A[i + 3] / B;
    }

    // 逐个转换回 half 并存储
    C[i + 0] = (half)results[0];
    C[i + 1] = (half)results[1];
    C[i + 2] = (half)results[2];
    C[i + 3] = (half)results[3];
}

__kernel void div_scalar_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const ushort B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    // 存根(stub)实现.
    return;
}

#endif // SUPPORTS_FP16