//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUSUMFUNC_HPP
#define CPUSUMFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
// #include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUsumFunction : public Op {
private:
    int thread_count = 4;
    Chl axis_;

public:
    CPUsumFunction(Backend *bn, string name, int threadCount, Chl axis) :
        Op(bn, name), thread_count(threadCount), axis_(axis) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int batch = inputs[0]->batch();
        int head = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();
        switch (axis_) {
        case BATCH:
            batch = 1;
            break;
        case HEAD:
            head = 1;
            break;
        case SEQUENCE:
            sequence = 1;
            break;
        case DIMENSION:
            dimension = 1;
            break;
        default:
            break;
        }
        outputs[0]->reshape(batch, head, sequence, dimension);
        outputs[0]->setDtype(inputs[0]->dtype());
        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int batch = inputs[0]->batch();
        int dim = inputs[0]->dimension();
        int seq = inputs[0]->sequence();
        int head = inputs[0]->head();
        switch (axis_) {
        case BATCH: {
            for (int h = 0; h < head; h++) {
                for (int s = 0; s < seq; ++s) {
                    for (int d = 0; d < dim; d++) {
                        float sum = 0.0f;
                        for (int n = 0; n < batch; n++) {
                            sum += inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        outputs[0]->setDataAt<float>(0, h, s, d, sum);
                    }
                }
            }
            break;
        }
        case HEAD: {
            for (int n = 0; n < batch; n++) {
                for (int s = 0; s < seq; ++s) {
                    for (int d = 0; d < dim; d++) {
                        float sum = 0.0f;
                        for (int h = 0; h < head; h++) {
                            sum += inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        outputs[0]->setDataAt<float>(n, 0, s, d, sum);
                    }
                }
            }
            break;
        }
        case SEQUENCE: {
            for (int n = 0; n < batch; n++) {
                for (int h = 0; h < head; h++) {
                    for (int d = 0; d < dim; d++) {
                        float sum = 0.0f;
                        for (int s = 0; s < seq; ++s) {
                            sum += inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        outputs[0]->setDataAt<float>(n, h, 0, d, sum);
                    }
                }
            }
            break;
        }
        case DIMENSION: {
            for (int n = 0; n < batch; n++) {
                for (int h = 0; h < head; h++) {
                    for (int s = 0; s < seq; s++) {
                        float sum = 0.0f;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            sum += inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        outputs[0]->setDataAt<float>(n, h, s, 0, sum);
                    }
                }
            }
            break;
        }
        default:
            break;
        }
        return MLLM_NO_ERROR;
    }
};

class CPUsumFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        Chl axis = (Chl)op_param.at("dim");
        return new CPUsumFunction(bn, name, threadCount, axis);
    }
};

} // namespace mllm
#endif // CPUSUMFUNC_HPP