//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUMATMULFUNC_HPP
#define CPUMATMULFUNC_HPP

#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "../compute/Matmul.hpp"
#include <cassert>
#include <vector>
#include <memory>
#include <algorithm> // For std::equal

namespace mllm {
class Tensor;

class CPUmmFunction : public Op {
private:
    int thread_count = 4;

    static void tranTensorChl(Tensor &input) {
        assert(input.ctype() == BSHD);
        auto b = input.batch();
        auto h = input.head();
        auto d = input.dimension();
        auto s = input.sequence();
        auto ori_seq_idx = input.chls()[SEQUENCE];
        auto ori_head_idx = input.chls()[HEAD];
        auto ori_dim_idx = input.chls()[DIMENSION];
        input.chls()[HEAD] = ori_seq_idx;
        input.chls()[DIMENSION] = ori_head_idx;
        input.chls()[SEQUENCE] = ori_dim_idx;
        input.changeCtype();
        input.reshape(b, h, s, d);
        input.transed() = true;
        input.undiffusion() = false;
        // if no TENSOR_STATIC_SHAPED
        if (input.masterTensor() != nullptr) {
            auto b_m = input.masterTensor()->batch();
            auto h_m = input.masterTensor()->head();
            auto d_m = input.masterTensor()->dimension();
            auto s_m = input.masterTensor()->sequence();
            input.masterTensor()->chls() = input.chls();
            input.masterTensor()->changeCtype();
            input.masterTensor()->reshape(b_m, h_m, s_m, d_m);
            for (auto &child : input.masterTensor()->childTensors()) {
                auto b_c = child->batch();
                auto h_c = child->head();
                auto d_c = child->dimension();
                auto s_c = child->sequence();
                child->chls() = input.chls();
                child->changeCtype();
                child->reshape(b_c, h_c, s_c, d_c);
            }
        } else {
            for (auto &child : input.childTensors()) {
                auto b_c = child->batch();
                auto h_c = child->head();
                auto d_c = child->dimension();
                auto s_c = child->sequence();
                child->chls() = input.chls();
                child->changeCtype();
                child->reshape(b_c, h_c, s_c, d_c);
            }
        }
    }

public:
    CPUmmFunction(Backend *bn, string name, int threadCount)
        : Op(bn, name), thread_count(threadCount) {}
    
   
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[1]->chls()[SEQUENCE] != 3) {
            tranTensorChl(*inputs[1]);
        }
        if (!inputs[1]->shape().empty() && !inputs[0]->shape().empty()) {
            assert(inputs[0]->dimension() == inputs[1]->sequence());
        }
        outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[1]->chls()[SEQUENCE] != 3) {
            tranTensorChl(*inputs[1]);
            assert(inputs[1]->chls()[SEQUENCE] == 3);
        }
        assert(inputs[0]->dimension() == inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }
    
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        bool isSame = std::equal(inputs[0]->chls().begin(), inputs[0]->chls().end(), inputs[1]->chls().begin());
        assert(inputs[0]->dtype() == MLLM_TYPE_F32);
        mat_mul(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, false, isSame, thread_count);
        return MLLM_NO_ERROR;
    }
};

class CPUmmFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUmmFunction(bn, name, threadCount);
    }
};

} // namespace mllm
#endif // CPUMATMULFUNC_HPP