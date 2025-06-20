//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPULIKEFUNC_HPP
#define CPULIKEFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <memory>

namespace mllm {
class Tensor;

class CPUlikeFunction : public Op {
private:
    int thread_count = 4;
    float like_value_;

public:
    CPUlikeFunction(Backend *bn, string name, int threadCount, float like_value)
        : Op(bn, name), thread_count(threadCount), like_value_(like_value) {}

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype()); // like_values
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // Allocate memory for the output tensor first
        outputs[0]->alloc();

        // Use a loop to correctly fill the tensor with the specified float value
        // as memset is for bytes and incorrect for arbitrary float values.
        auto* out_ptr = outputs[0]->hostPtr<float>();
        int count = outputs[0]->count();

        #pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < count; ++i) {
            out_ptr[i] = like_value_;
        }
        
        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUlikeFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam contains the key "like_value"
        float like_value = op_param.at("like_value");
        return new CPUlikeFunction(bn, name, threadCount, like_value);
    }
};

} // namespace mllm
#endif // CPULIKEFUNC_HPP