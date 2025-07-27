// 文件名: kernel/mul.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ==================================================================
// 1. Tensor * Tensor Multiplication Kernels
// ==================================================================

__kernel void mul_float(
    __global const float *A,
    __global const float *B,
    __global float *C) {
    size_t index = get_global_id(0);
    C[index] = A[index] * B[index];
}

__kernel void mul_float_image2d(
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
    float4 result = inA * inB;
    write_imagef(output, pos, result);
}

__kernel void mul_fp16_vector(
    __global const half *A,
    __global const half *B,
    __global half *C) {
    const int i = get_global_id(0);
    half4 a_vec = vload4(i, A);
    half4 b_vec = vload4(i, B);
    half4 c_vec = a_vec * b_vec;
    vstore4(c_vec, i, C);
}

__kernel void mul_fp16_image2d(
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
    half4 result = inA * inB;
    write_imageh(output, pos, result);
}

// ==================================================================
// 2. Tensor * Scalar Multiplication Kernels
// ==================================================================

__kernel void mul_scalar_float(
    __global const float *A,
    const float B,
    __global float *C) {
    size_t index = get_global_id(0);
    C[index] = A[index] * B;
}

__kernel void mul_scalar_fp16_vector(
    __global const half *A,
    const half B,
    __global half *C) {
    const int i = get_global_id(0);
    half4 a_vec = vload4(i, A);
    half4 b_vec = (half4)(B);
    half4 c_vec = a_vec * b_vec;
    vstore4(c_vec, i, C);
}

__kernel void mul_scalar_float_image2d(
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
    float4 result = inA * inB;
    write_imagef(output, pos, result);
}

__kernel void mul_scalar_fp16_image2d(
    sampler_t sampler,
    __read_only image2d_t inputA,
    const half B,
    __write_only image2d_t output,
    const int width,
    const int height) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x >= width || pos.y >= height) { return; }
    half4 inA = read_imageh(inputA, sampler, pos);
    half4 inB = (half4)(B);
    half4 result = inA * inB;
    write_imageh(output, pos, result);
}