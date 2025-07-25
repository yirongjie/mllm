#ifndef OPENCL_ADD_FUNC_OP_HPP
#define OPENCL_ADD_FUNC_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLAddFuncOp : public Op {
public:
    OpenCLAddFuncOp(Backend *bn, std::string name);
    ~OpenCLAddFuncOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    // cl_kernel buffer_kernel_ = nullptr; 
    // cl_kernel image_kernel_ = nullptr;  // 新的 Image2D kernel

    cl_kernel kernel_fp32_buffer_ = nullptr;
    cl_kernel kernel_fp32_image_ = nullptr;
    cl_kernel kernel_fp16_buffer_ = nullptr;
    cl_kernel kernel_fp16_image_ = nullptr;
    
    cl_sampler sampler_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

// OpenCLAddFuncOp 的创建器，用于工厂模式
class OpenCLAddFuncOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // 对于简单的加法，我们可能不需要 op_param 和 threadCount
        return new OpenCLAddFuncOp(bn, name);
    }
};

} // namespace mllm

#endif // OPENCL_ADD_FUNC_OP_HPP