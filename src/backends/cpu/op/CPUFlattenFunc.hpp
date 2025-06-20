//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUFLATTENFUNC_HPP
#define CPUFLATTENFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <iostream>
#include <vector>

namespace mllm {
class Tensor;

class CPUflattenFunction : public Op {
private:
    int thread_count = 4;
    Chl axis_start_;
    Chl axis_end_;

public:
    CPUflattenFunction(Backend *bn, string name, int threadCount, Chl axis_start, Chl axis_end)
        : Op(bn, name), thread_count(threadCount), axis_start_(axis_start), axis_end_(axis_end) {}

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];

        int dim_b = input->batch();
        int dim_h = 0;
        int dim_s = 0;
        int dim_d = 0;

        if (input->shape().size() == 4) {
            dim_h = input->head();
            dim_s = input->sequence();
            dim_d = input->dimension();
            if (axis_start_ == BATCH && axis_end_ == SEQUENCE) {
                dim_b = 1;
                dim_s = input->sequence() * input->batch();
            } else if (axis_start_ == HEAD && axis_end_ == SEQUENCE) {
                dim_h = 1;
                dim_s = input->sequence() * input->head();
            } else if (axis_start_ == HEAD && axis_end_ == DIMENSION) {
                dim_h = 1;
                dim_d = input->dimension() * input->head();
            } else {
                // This combination might not be a simple flatten, but a transpose+flatten.
                // Assuming it implies a view can be created.
            }
        } else if (input->shape().size() == 5) {
            // Logic from original code for 5D tensors
            if (axis_start_ == CHANNLE && axis_end_ == HEIGHT) {
                dim_h = 1;
                dim_s = input->channel() * input->height() * input->time();
                dim_d = input->width();
            } else if (axis_start_ == HEIGHT && axis_end_ == CHANNLE) {
                dim_h = 1;
                dim_s = input->channel() * input->height() * input->width();
                dim_d = input->time();
            }
        }
        assert(dim_b > 0 || dim_h > 0 || dim_s > 0 || dim_d > 0);

        // This is a metadata-only operation. The output becomes a view of the input.
        // It points to the same data but has a different shape.
        output->shallowCopyFrom(input.get());
        output->reshape(dim_b, dim_h, dim_s, dim_d);

        if (input->ctype() == BCTHW) { // Propagate channel information if applicable
            output->chls()[BATCH] = 0;
            output->chls()[SEQUENCE] = 1;
            output->chls()[HEAD] = 2;
            output->chls()[DIMENSION] = 3;
            output->setCtype(BSHD);
        }
        
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // No data movement needed, all work done in reshape by creating a view.
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs)  override {
        // inputs[0]->shallowCopyFrom(outputs[0].get(), false);
        // Chl axis_start = (Chl)args[0];
        // Chl axis_end = (Chl)args[1];
        if ((axis_start_ == TIME & axis_end_ == WIDTH && inputs[0]->ctype() == BCTHW)
            || (axis_start_ == CHANNLE & axis_end_ == HEIGHT && inputs[0]->ctype() == BWCTH)
            || (axis_start_ == HEIGHT & axis_end_ == CHANNLE && inputs[0]->ctype() == BTHWC)
            || (axis_start_ == BATCH & axis_end_ == SEQUENCE && inputs[0]->ctype() != BCTHW)
            || (axis_start_ == HEAD & axis_end_ == SEQUENCE && inputs[0]->ctype() == BSHD)
            || (axis_start_ == HEAD & axis_end_ == SEQUENCE && inputs[0]->ctype() == BHDS)
            || (axis_start_ == HEAD & axis_end_ == DIMENSION && inputs[0]->ctype() == BSHD)
            || (axis_start_ == HEAD & axis_end_ == DIMENSION && inputs[0]->ctype() == BHDS)
            || (axis_start_ == HEAD & axis_end_ == SEQUENCE && inputs[0]->ctype() == BDSH)) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->shallowCopyFrom(outputs[0].get(), false);
        // } else if (inputs[0]->module()->op_transposed_flag) {
        //     if (inputs[0]->masterTensor() == nullptr) {
        //         inputs[0]->free();
        //     }
        //     outputs[0]->setDtype(inputs[0]->dtype());
        //     outputs[0]->alloc();
        //     inputs[0]->shallowCopyFrom(outputs[0].get(), false);
        } else {
            std::cout << "[TODO]Tensor.Flatten not support!!!!" << std::endl;
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUflattenFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam contains keys "axis_start" and "axis_end"
        Chl axis_start = (Chl)op_param.at("axis_start");
        Chl axis_end = (Chl)op_param.at("axis_end");
        return new CPUflattenFunction(bn, name, threadCount, axis_start, axis_end);
    }
};

} // namespace mllm
#endif // CPUFLATTENFUNC_HPP