//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUCATFUNC_HPP
#define CPUCATFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <algorithm>
#include <cassert>
#include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUcatFunction : public Op {
private:
    int thread_count = 4;
    Chl axis_;

public:
    CPUcatFunction(Backend *bn, string name, int threadCount, Chl axis)
        : Op(bn, name), thread_count(threadCount), axis_(axis) {}
    
    // 注意：保留 setUp 函数。
    // Op 的执行引擎(如 runLayer)也需要相应修改来调用此函数，才能使其中逻辑生效。
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (outputs[0]->shape().empty()) {
            int expd_batch_ = inputs[0]->batch();
            for (int ii = 0; ii < inputs.size(); ++ii) {
                auto input = inputs[ii];
                expd_batch_ = std::max(input->batch(), expd_batch_);
            }
            int dim_b = expd_batch_;
            int dim_h = inputs[0]->head();
            int dim_s = inputs[0]->sequence();
            int dim_d = inputs[0]->dimension();
            
            if (axis_ == BATCH) {
                dim_b = 0;
                for (auto& input : inputs) dim_b += input->batch();
            } else if (axis_ == HEAD) {
                dim_h = 0;
                for (auto& input : inputs) dim_h += input->head();
            } else if (axis_ == SEQUENCE) {
                dim_s = 0;
                for (auto& input : inputs) dim_s += input->sequence();
            } else if (axis_ == DIMENSION) {
                dim_d = 0;
                for (auto& input : inputs) dim_d += input->dimension();
            }

            outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
        }

        if (axis_ == HEAD) {
            int cbatch = 0;
            int chead = 0;
            int cseq = 0;
            int cdim = 0;
            if (inputs.size() > 1 && inputs[0]->hostPtr<float>() == inputs[1]->hostPtr<float>()) {
                if (inputs[0]->masterTensor() == nullptr) {
                    inputs[0]->free();
                }
                inputs[0]->shallowCopyFrom(outputs[0].get(), false, {cbatch, chead, cseq, cdim});
            } else {
                for (int idx = 0; idx < inputs.size(); idx++) {
                    if (inputs[idx]->masterTensor() == nullptr) {
                        inputs[idx]->free();
                    }
                    if (idx > 0) {
                        chead += inputs[idx - 1]->head();
                    }
                    inputs[idx]->shallowCopyFrom(outputs[0].get(), false, {cbatch, chead, cseq, cdim}); // b,h,s,d
                }
            }
        } else if (axis_ == SEQUENCE && inputs[0]->head() != 1) {
            int cbatch = 0;
            int chead = 0;
            int cseq = 0;
            int cdim = 0;
            for (int idx = 0; idx < inputs.size(); idx++) {
                if (inputs[idx]->masterTensor() == nullptr) {
                    inputs[idx]->free();
                }
                if (idx > 0) {
                    cseq += inputs[idx - 1]->sequence();
                }
                inputs[idx]->shallowCopyFrom(outputs[0].get(), false, {cbatch, chead, cseq, cdim}); // b,h,s,d
            }
        } else if (axis_ == DIMENSION && inputs[0]->head() != 1) {
            int cbatch = 0;
            int chead = 0;
            int cseq = 0;
            int cdim = 0;
            for (int idx = 0; idx < inputs.size(); idx++) {
                if (inputs[idx]->masterTensor() == nullptr) {
                    inputs[idx]->free();
                }
                if (idx > 0) {
                    cdim += inputs[idx - 1]->dimension();
                }
                int tmp_agg_idx = -1;
                if (inputs[idx]->deaggregatedTensor() != nullptr) {
                    for (int t = 0; t < inputs[idx]->deaggregatedTensor()->aggregatedTensors().size(); t++) {
                        if (inputs[idx]->deaggregatedTensor()->aggregatedTensors()[t] == inputs[idx]) {
                            tmp_agg_idx = t;
                            break;
                        }
                    }
                }
                inputs[idx]->shallowCopyFrom(outputs[0].get(), false, {cbatch, chead, cseq, cdim}); // b,h,s,d
                if (inputs[idx]->deaggregatedTensor() != nullptr && tmp_agg_idx != -1) {
                    inputs[idx]->deaggregatedTensor()->aggregatedTensors()[tmp_agg_idx] = inputs[idx];
                }
            }
        }
        return MLLM_NO_ERROR;
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int expd_batch_ = inputs[0]->batch();
        for (int ii = 0; ii < inputs.size(); ++ii) {
            auto input = inputs[ii];
            expd_batch_ = std::max(input->batch(), expd_batch_);
        }
        int dim_b = expd_batch_;
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();

        if (axis_ == BATCH) {
            dim_b = 0;
            for (auto& input : inputs) dim_b += input->batch();
        } else if (axis_ == HEAD) {
            dim_h = 0;
            for (auto& input : inputs) dim_h += input->head();
        } else if (axis_ == SEQUENCE) {
            dim_s = 0;
            for (auto& input : inputs) dim_s += input->sequence();
        } else if (axis_ == DIMENSION) {
            dim_d = 0;
            for (auto& input : inputs) dim_d += input->dimension();
        }
        
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        if (outputs[0]->masterTensor() == nullptr) {
            outputs[0]->setDtype(inputs[0]->dtype());
            // 遵从原始 reshape 逻辑，在这里 alloc
            outputs[0]->alloc();
        }
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int expd_batch_ = inputs[0]->batch();
        int expd_batch_input_idx = 0;
        for (int ii = 0; ii < inputs.size(); ++ii) {
            auto input = inputs[ii];
            if (input->batch() > expd_batch_) {
                expd_batch_ = input->batch();
                expd_batch_input_idx = ii;
            }
        }

        if (axis_ == BATCH) {
            int batch_offset = 0;
            for (int n = 0; n < inputs.size(); ++n) {
                auto input = inputs[n];
                size_t copy_size = (size_t)input->batch() * input->head() * input->sequence() * input->dimension() * input->dtypeSize();
                memcpy(outputs[0]->ptrAt<char>(batch_offset, 0, 0, 0),
                       input->ptrAt<char>(0, 0, 0, 0),
                       copy_size);
                batch_offset += input->batch();
            }
        } else if (axis_ == DIMENSION) {
            for (int n = 0; n < expd_batch_; ++n) {
                for (int c = 0; c < inputs[0]->head(); ++c) {
                    for (int h = 0; h < inputs[0]->sequence(); ++h) {
                        int w = 0;
                        for (int idx = 0; idx < inputs.size(); idx++) {
                            int dim_size = inputs[idx]->dimension();
                            auto n_ = (idx == expd_batch_input_idx) ? n : 0;
                            size_t copy_size = (size_t)dim_size * inputs[idx]->dtypeSize();
                            memcpy(outputs[0]->ptrAt<char>(n, c, h, w),
                                   inputs[idx]->ptrAt<char>(n_, c, h, 0),
                                   copy_size);
                            w += dim_size;
                        }
                    }
                }
            }
        } else if (axis_ == SEQUENCE) {
            for (int n = 0; n < expd_batch_; ++n) {
                for (int h = 0; h < outputs[0]->head(); ++h) {
                    if (inputs[0]->ctype() == BSHD || inputs[0]->head() == 1) {
                        int s_base = 0;
                        for (int idx = 0; idx < inputs.size(); idx++) {
                            auto n_ = (idx == expd_batch_input_idx) ? n : 0;
                            size_t copy_size = (size_t)inputs[idx]->sequence() * inputs[idx]->dimension() * inputs[idx]->dtypeSize();
                            memcpy(outputs[0]->ptrAt<char>(n, h, s_base, 0),
                                   inputs[idx]->ptrAt<char>(n_, h, 0, 0),
                                   copy_size);
                            s_base += inputs[idx]->sequence();
                        }
                    } else if (inputs[0]->ctype() == BHDS) {
                        int s_base = 0;
                        for (int idx = 0; idx < inputs.size(); idx++) {
                            auto n_ = (idx == expd_batch_input_idx) ? n : 0;
                            for (int d = 0; d < inputs[idx]->dimension(); ++d) {
                                size_t copy_size = (size_t)inputs[idx]->sequence() * inputs[idx]->dtypeSize();
                                memcpy(outputs[0]->ptrAt<char>(n, h, s_base, d),
                                       inputs[idx]->ptrAt<char>(n_, h, 0, d),
                                       copy_size);
                            }
                            s_base += inputs[idx]->sequence();
                        }
                    }
                }
            }
        } else if (axis_ == HEAD) {
            if (inputs.size() > 1 && inputs[0]->hostPtr<char>() == inputs[1]->hostPtr<char>()) {
                for (int b = 0; b < outputs[0]->batch(); ++b) {
                    for (int s = 0; s < inputs[0]->sequence(); ++s) {
                        for (int h_ = 1; h_ < outputs[0]->head(); ++h_) {
                            size_t copy_size = (size_t)inputs[0]->dimension() * inputs[0]->dtypeSize();
                            memcpy(outputs[0]->ptrAt<char>(b, h_, s, 0),
                                   outputs[0]->ptrAt<char>(b, 0, s, 0),
                                   copy_size);
                        }
                    }
                }
            } else {
                for (int b = 0; b < expd_batch_; ++b) {
                    #pragma omp parallel for num_threads(thread_count)
                    for (int s = 0; s < inputs[0]->sequence(); ++s) {
                        int head_offset = 0;
                        for (int idx = 0; idx < inputs.size(); idx++) {
                            auto b_ = (idx == expd_batch_input_idx) ? b : 0;
                            size_t copy_size = (size_t)inputs[idx]->dimension() * inputs[idx]->head() * inputs[idx]->dtypeSize();
                            memcpy(outputs[0]->ptrAt<char>(b, head_offset, s, 0),
                                   inputs[idx]->ptrAt<char>(b_, 0, s, 0),
                                   copy_size);
                            head_offset += inputs[idx]->head();
                        }
                    }
                }
            }
        }
        return MLLM_NO_ERROR;
    }
};

class CPUcatFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        Chl axis = (Chl)op_param.at("axis");
        return new CPUcatFunction(bn, name, threadCount, axis);
    }
};

} // namespace mllm
#endif // CPUCATFUNC_HPP