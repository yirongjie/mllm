
#ifndef MLLM_CPUKVCACHE_H
#define MLLM_CPUKVCACHE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"
#include "ParamLoader.hpp"

namespace mllm {

class CPUKVCache final : public Op {
public:
    CPUKVCache(Backend *bn, string opName, int hidden, int head, int n_rep, bool fa2, int cache_max = 100, int threadCount = 4);
    virtual ~CPUKVCache() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    shared_ptr<Tensor> cache_;

    int getCacheSeqLen() override {
        return cache_seq_len_;
    }
    void clearCache() override {
        cache_seq_len_ = 0;
        cache_->cache_seq_len_ = cache_seq_len_;
    }

    void setForXnn(bool for_xnn) {
        for_xnn_ = for_xnn;
    }

    ErrorCode updateVerifiedKVCache(const std::vector<unsigned int> &verified_position_ids);

private:
    int thread_count = 4;

    int cache_seq_len_ = -999;
    int n_rep_ = 1;

    bool for_xnn_ = false;
    int cache_limit_;

    bool fa2_ = false; // not_fa2
};

class CPUKVCacheCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int n_rep = (int)op_param["n_rep"];
        int cache_max = (int)op_param["cache_max"];
        bool for_xnn = (bool)op_param["for_xnn"];
        int hidden = (int)op_param["hidden"];
        int head = (int)op_param["head"];
        bool fa2 = (bool)op_param["fa2"];
        auto ret = new CPUKVCache(bn, name, hidden, head, n_rep, fa2, cache_max, threadCount);
        ret->setForXnn(for_xnn);
        return ret;
    }
};

} // namespace mllm

#endif // MLLM_CPUKVCACHE_H