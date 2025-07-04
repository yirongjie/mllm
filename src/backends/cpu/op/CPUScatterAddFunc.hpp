//
// Created by Rongjie Yi on 24-12-26.
//

#ifndef CPUSCATTEADDFUNC_HPP
#define CPUSCATTEADDFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include "../compute/Arithmetic.hpp"
#include <memory>

namespace mllm {
class Tensor;

class CPUScatterAddFunction : public Op {
private:
    int thread_count = 4;

public:
    CPUScatterAddFunction(Backend *bn, string name, int threadCount) :
        Op(bn, name), thread_count(threadCount) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[1]->batch() == 0) {
            return MLLM_NO_ERROR;
        }
        assert(inputs.size() == 3);
        assert(inputs[0]->batch() == 1);
        assert(inputs[0]->head() == 1);
        auto dest_input = inputs[0];
        auto src_input = inputs[1];
        auto replace_idx = inputs[2];
        assert(replace_idx->batch() == 1);
        assert(replace_idx->sequence() == 1);
        assert(replace_idx->head() == 1);
        // #pragma omp parallel for num_threads(CPUBackend::cpu_threads)
        for (int r_idx = 0; r_idx < replace_idx->dimension(); r_idx++) {
            auto replace_seq = (int)replace_idx->dataAt<float>(0, 0, 0, r_idx);
            auto dst_ptr = dest_input->ptrAt<float>(0, 0, replace_seq, 0);
            auto src_ptr = src_input->ptrAt<float>(0, 0, r_idx, 0);
            // memcpy(dst_ptr, src_ptr, sizeof(float) * src_input->dimension());
            float tmp[src_input->dimension()];
            memcpy(tmp, dst_ptr, sizeof(float) * dest_input->dimension());
            mllm_add_fp32(tmp,
                          src_ptr,
                          dst_ptr, dest_input->dimension());
        }
        return MLLM_NO_ERROR;
    }
};

class CPUScatterAddFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUScatterAddFunction(bn, name, threadCount);
    }
};

} // namespace mllm
#endif // CPUSCATTEADDFUNC_HPP