#ifndef MLLM_CPUROPE_H
#define MLLM_CPUROPE_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPURoPE final : public Op {
public:
    CPURoPE(Backend *bn, string opName, int pose_type, int threadCount);
    CPURoPE(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings, int threadCount);
    virtual ~CPURoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    //    Tensor freq_;
    // static Tensor sin_;
    // static Tensor cos_;
    static vector<vector<float>> sin_;
    static vector<vector<float>> cos_;
    static int global_pose_type_;
    static int ishape_old;
    int rope_theta_ = 10000;
    int h_cnt_ = 0;
    int pos_max_ = 16384;
    int pose_type_ = 4;
    int ishape;
    int thread_count = 4;
};

class CPURoPECreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int pose_type = op_param["pose_type"];
        if (op_param.find("rope_theta") == op_param.end()) {
            return new CPURoPE(bn, name, pose_type, threadCount);
        }
        float rope_theta = op_param["rope_theta"];
        int max_position_embeddings = op_param["max_position_embeddings"];
        return new CPURoPE(bn, name, pose_type, rope_theta, max_position_embeddings, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUROPE_H