//
// Created by Xiang Li on 2023/12/16.
//

// #ifdef ANDROID_API

#include "LibHelper.hpp"
#include <Types.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "Generate.hpp"
#include "models/bert/configuration_bert.hpp"
#include "models/bert/modeling_bert.hpp"
#include "models/bert/tokenization_bert.hpp"
#include "models/fuyu/configuration_fuyu.hpp"
#include "models/fuyu/modeling_fuyu.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl.hpp"
#include "models/qwen2_vl/processing_qwen2_vl.hpp"
#include "models/phonelm/configuration_phonelm.hpp"
#include "models/phonelm/modeling_phonelm.hpp"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "models/smollm/tokenization_smollm.hpp"
#include "tokenizers/Unigram/Unigram.hpp"
#include "models/fuyu/processing_fuyu.hpp"
#include "processor/PostProcess.hpp"
using namespace mllm;

#ifdef USE_QNN
#include "models/qwen/modeling_qwen_npu.hpp"
#include "models/phonelm/modeling_phonelm_npu.hpp"
#include "models/qwen2_vl/modeling_qwen2_vl_npu.hpp"
#endif
inline bool exists_test(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

unsigned int LibHelper::postProcessing(shared_ptr<Tensor> result, shared_ptr<Tensor> &out_result) const {
    // switch (model_) {
    // case LLAMA: {
    //     return 0;
    // }
    // case FUYU: {
    //     // return chatPostProcessing(unsigned int token_idx, Tensor &tokens_tensor, const int &clean_tensors);
    // }
    // default: return 0;
    // }
    return 0;
}

bool LibHelper::setUp(const std::string &base_path, std::string weights_path, std::string qnn_weights_path, std::string vocab_path, std::string merge_path, PreDefinedModel model, MLLMBackendType backend_type) {
    FuyuConfig fuyuconfig(tokens_limit, "8B");
    QWenConfig qwconfig(tokens_limit, "1.5B");
    string qwvl_b = "1.5b";
#ifdef USE_QNN
    if (backend_type == MLLMBackendType::QNN) {
        qwvl_b = "1.5b-rotated";
        LOGI("initBackend");
        Module::initBackend(MLLM_QNN);
        LOGI("initBackend");
    }
#endif
    LOGI("Qwen2VLConfig qwvlconfig: %s", qwvl_b.c_str());
    Qwen2VLConfig qwvlconfig(tokens_limit, qwvl_b);
    qwvlconfig.attn_implementation = "eager";
    BertConfig bertconfig;
    PhoneLMConfig phone_config(tokens_limit, "1.5B");
    vocab_path = base_path + vocab_path;
    merge_path = base_path + merge_path;
    weights_path = base_path + weights_path;
    qnn_weights_path = base_path + qnn_weights_path;
    model_ = model;
    backend_ = backend_type;

    LOGI("Loading qnn model from %s", qnn_weights_path.c_str());
    LOGI("Loading model from %s", weights_path.c_str());

    switch (model) {
    case QWEN25:
        qwconfig = QWenConfig(tokens_limit, "1.5B");
        tokenizer_ = make_shared<QWenTokenizer>(vocab_path, merge_path);
        module_ = make_shared<QWenForCausalLM>(qwconfig);
        break;
    case QWEN15:
        qwconfig = QWenConfig(tokens_limit, "1.8B");
        tokenizer_ = make_shared<QWenTokenizer>(vocab_path, merge_path);
        module_ = make_shared<QWenForCausalLM>(qwconfig);
#ifdef USE_QNN
        if (backend_type == MLLMBackendType::QNN) {
            int chunk_size = 64;
            prefill_module_ = make_shared<v2::QWenForCausalLM_NPU>(qwconfig, chunk_size);
            prefill_module_->load(qnn_weights_path);

            auto tokenizer = dynamic_pointer_cast<QWenTokenizer>(tokenizer_);
            // warmup START
            std::string input_str = " ";
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 151936);
            auto input_tensor = res.second;
            auto real_seq_length = res.first;
            LlmTextGeneratorOpts opt{
                .max_new_tokens = 1,
                .do_sample = false,
                .is_padding = true,
                .seq_before_padding = real_seq_length,
                .chunk_size = chunk_size,
            };
            prefill_module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_string);
                if (!not_end) { return false; }
                return true;
            });
            Module::isFirstChunk = false;
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
            Module::isMultiChunkPrefilling = true;
            // warmup END
            LOGE("QNN Warmup finished.");
        }
#endif
        break;

    case FUYU:
        processor_ = new FuyuProcessor(vocab_path, 224, 224);
        module_ = make_shared<FuyuModel>(fuyuconfig);
        break;
    case QWEN2VL:
        processor_ = new Qwen2VLProcessor(vocab_path, merge_path);
        LOGI("Init Qwen2VLProcessor: %d", backend_type);
#ifdef USE_QNN
        if (backend_type == MLLMBackendType::QNN) {
            int chunk_size = 256;
            prefill_module_ = make_shared<Qwen2VL_PrefillBody>(qwvlconfig, chunk_size);
            prefill_module_->load(qnn_weights_path);
            prefill_embedding_ = make_shared<Qwen2VL_ImagePatchAndEmbedding>(qwvlconfig);
            prefill_embedding_->load(weights_path);
            qwvlconfig.attn_implementation = "eager";
            module_ = make_shared<Qwen2VL_Decoding_Model>(qwvlconfig);
        } else {
#endif
            module_ = make_shared<Qwen2VLModel>(qwvlconfig);

#ifdef USE_QNN
        }
#endif
        break;
    case Bert:
        tokenizer_ = make_shared<BertTokenizer>(vocab_path, true);
        module_ = make_shared<BertModel>(bertconfig);
        break;

    case PhoneLM:
        tokenizer_ = make_shared<SmolLMTokenizer>(vocab_path, merge_path);
        module_ = make_shared<PhoneLMForCausalLM>(phone_config);
#ifdef USE_QNN
        if (backend_type == MLLMBackendType::QNN) {
            prefill_module_ = make_shared<PhoneLMForCausalLM_NPU>(phone_config);
            prefill_module_->load(qnn_weights_path);

            auto tokenizer = dynamic_pointer_cast<SmolLMTokenizer>(tokenizer_);
            // warmup START
            std::string input_str = " ";
            int chunk_size = 64;
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 49152);
            auto input_tensor = res.second;
            auto real_seq_length = res.first;
            LlmTextGeneratorOpts opt{
                .max_new_tokens = 1,
                .do_sample = false,
                .is_padding = true,
                .seq_before_padding = real_seq_length,
                .chunk_size = chunk_size,
            };
            prefill_module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_string);
                if (!not_end) { return false; }
                return true;
            });
            Module::isFirstChunk = false;
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
            Module::isMultiChunkPrefilling = true;
            // warmup END
            LOGE("QNN Warmup finished.");
        }
#endif
        break;
    }
    module_->load(weights_path);
    is_first_run_cond_ = true;

    return true;
}

void LibHelper::setCallback(callback_t callback) {
    this->callback_ = std::move(callback);
}

void LibHelper::run(std::string &input_str, uint8_t *image, unsigned max_step, unsigned int image_length, bool chat_template) {
    std::string output_string_;
    LOGE("Running model %d", model_);
    unsigned max_new_tokens = 500;
    LOGE("Running backend %d", backend_);
    vector<double> profiling_data(3);

    if (model_ == QWEN15 || model_ == QWEN25) {
        auto tokenizer = dynamic_pointer_cast<QWenTokenizer>(tokenizer_);
        if (chat_template) input_str = tokenizer_->apply_chat_template(input_str);
        if (backend_ == MLLMBackendType::QNN) {
            int chunk_size = 64;
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 151936);
            auto input_tensor = res.second;
            max_new_tokens = tokens_limit - input_tensor.sequence();
            auto real_seq_length = res.first;
            const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
            const int chunk_num = seq_length_padding / chunk_size;
            bool isSwitched = false;

            // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setTotalSequenceLength(real_seq_length);
            // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setChunkSize(chunk_size);

            LlmTextGeneratorOpts opt{
                .max_new_tokens = 1,
                .do_sample = false,
                .is_padding = true,
                .seq_before_padding = real_seq_length,
                .chunk_size = chunk_size,
            };
            std::vector<Tensor> chunked_tensors(chunk_num);
            for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
                chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU].get());
                chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
                chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
                chunked_tensors[chunk_id].setName("input-chunk-" + to_string(chunk_id));
                chunked_tensors[chunk_id].shallowCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

                prefill_module_->generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                    if (!isSwitched && chunk_id == 0 && static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->isStageSwitching()) {
                        // turn off switching at the first chunk of following inputs
                        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
                        isSwitched = true;
                    }
                    // switch_flag = true;
                    auto out_string = tokenizer_->detokenize({out_token});
                    auto [not_end, output_string] = tokenizer_->postprocess(out_string);
                    if (chunk_id == chunk_num - 1) { // print the output of the last chunk
                        output_string_ += output_string;
                        if (!not_end) {
                            auto profile_res = prefill_module_->profiling("Prefilling");
                            if (profile_res.size() == 3) {
                                profiling_data[0] += profile_res[0];
                                profiling_data[1] = profile_res[1];
                            }
                            callback_(output_string_, !not_end, profiling_data);
                        }
                        callback_(output_string_, !not_end, {});
                    }
                    return true;
                });
                Module::isFirstChunk = false;
            }
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(AUTOREGRESSIVE);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();

            opt = LlmTextGeneratorOpts{
                .max_new_tokens = max_new_tokens - 1,
                .do_sample = false,
                .temperature = 0.3f,
                .top_k = 50,
                .top_p = 0.f,
                .is_padding = false,
            };
            isSwitched = false;
            module_->generate(chunked_tensors.back(), opt, [&](unsigned int out_token) -> bool {
                if (!isSwitched) {
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
                    isSwitched = true;
                }
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data[0] += profile_res[0];
                        profiling_data[2] = profile_res[2];
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
        } else { // CPU
            auto input_tensor = tokenizer_->tokenize(input_str);
            max_new_tokens = tokens_limit - input_tensor.sequence();
            LlmTextGeneratorOpts opt{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
            };
            module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data = profile_res;
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            module_->clear_kvcache();
        }

    } else if (model_ == FUYU) {
        auto processor = dynamic_cast<FuyuProcessor *>(processor_);
        auto input_tensors = processor->process(input_str, {image}, {image_length});
        for (int step = 0; step < 100; step++) {
            auto result = (*module_)({input_tensors[0], input_tensors[1], input_tensors[2]});
            auto outputs = processor->detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [end, string] = processor->postprocess(out_string);
            output_string_ += string;
            callback_(output_string_, !end, {});
            if (!end) { break; }
            chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});
        }
        module_->clear_kvcache();
    } else if (model_ == QWEN2VL) {
        auto processor = dynamic_cast<Qwen2VLProcessor *>(processor_);
        input_str = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1.<|vision_start|><|image_pad|><|vision_end|>" + input_str;
        input_str = processor->tokenizer->apply_chat_template(input_str);
        auto input_tensors = processor->process(input_str, {image}, {image_length});
        LOGE("Instruct:  %s", input_str.c_str());
        LOGE("Tokens:  %d", input_tensors[0].sequence());

#ifdef USE_QNN
        if (backend_ == MLLMBackendType::QNN) {
            int chunk_size = 256;

            const int real_seq_length = input_tensors[0].sequence();
            const int num_iter = (real_seq_length + chunk_size - 1) / chunk_size;
            auto model = dynamic_cast<Qwen2VL_Decoding_Model *>(module_.get());
            auto prefill_embedding = dynamic_cast<Qwen2VL_ImagePatchAndEmbedding *>(prefill_embedding_.get());
            // padding the position_ids to total chunk length(example: 256*2) for CPUMultimodalRoPEPipeline
            LOGE("before get_position_ids");
            prefill_embedding->get_position_ids(input_tensors, chunk_size * num_iter);
            LOGE("after get_position_ids");

            // warm up (still need a warm up as the setup stage is not omitted now)
            auto merged_embd_warmup_tensor = Tensor(Backend::global_backends[MLLM_QNN]);
            merged_embd_warmup_tensor.reshape(1, 1, chunk_size, 1536);
            merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
            merged_embd_warmup_tensor.alloc();
            merged_embd_warmup_tensor.setTtype(INPUT_TENSOR);
            input_tensors.back().setTtype(INPUT_TENSOR);
            vector<Tensor> prefill_input = {merged_embd_warmup_tensor, input_tensors.back()};
            (*prefill_module_)(prefill_input);
            LOGE("after warm up");

            Module::isFirstChunk = false;
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();

            // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setTotalSequenceLength(real_seq_length);
            // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setChunkSize(chunk_size);

            for (auto &t : input_tensors) {
                t.setTtype(INPUT_TENSOR);
            }

            // 1. get the vit embedding using CPU
            auto merged_embd = (*prefill_embedding)(input_tensors);
            LOGE("after vit embedding");

            // free prefill embedding tensor, approximately free 1GB for 59ms
            auto begin_free = mllm_time_ms();
            auto &embedding_act = prefill_embedding->activation_tensors;
            // go through the activation tensors to get the merged_embd
            for (auto iter = embedding_act.begin(); iter != embedding_act.end(); ++iter) {
                // std::cout << iter->first << std::endl;
                if (iter->first.find("input") != std::string::npos || iter->first.find("index_put") != std::string::npos) {
                    continue;
                }
                iter->second->free();
            }
            auto end_free = mllm_time_ms();
            LOGE("after free");

            // 2. QNN LLM Prefill
            unsigned int out_token = 0;
            for (auto i = 0; i < num_iter; ++i) {
                // copy the data from merged_embd[0] to merged_embd_warmup_tensor
                auto source = merged_embd[0].ptrAt<float>(0, 0, chunk_size * i, 0);
                auto dest = prefill_input[0].hostPtr<void>();
                if (i == 0) {
                    memcpy(dest, source, prefill_input[0].cntSize());
                }
                {
                    memcpy(dest, source, (merged_embd[0].sequence() % chunk_size) * merged_embd[0].dimension() * sizeof(float));
                }

                auto result = (*prefill_module_)(prefill_input);

                if (i == 0) { // turn off switching to avoid RoPE h_cnt_ reset to curSequenceLength in next chunk
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
                }

                if (i == 1) {
                    auto outputs = processor->detokenize(result[0], real_seq_length % chunk_size);
                    auto out_string = outputs.first;
                    out_token = outputs.second;
                    // auto [not_end, output_string] = processor->tokenizer->postprocess(out_string);
                    // std::cout << output_string << std::flush;
                    auto [end, string] = processor->tokenizer->postprocess(out_string);
                    output_string_ += string;
                    callback_(output_string_, !end, {});
                }
            }

            chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});

            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(AUTOREGRESSIVE);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();

            // 3. CPU LLM Decoding
            for (auto &t : input_tensors) { // set to INPUT_TENSOR to let decoding module update act
                t.setTtype(INPUT_TENSOR);
            }

            const int last_position_id = input_tensors[3].dataAt<float>(0, 0, 0, real_seq_length - 1);
            for (int step = 0; step < 100; step++) {
                // use the last position id(no padding position) in decoding
                prefill_embedding->get_position_ids(input_tensors, 0, last_position_id + 1 + step);

                auto result = (*model)(input_tensors);
                auto outputs = processor->detokenize(result[0]);
                auto out_string = outputs.first;
                auto out_token = outputs.second;
                auto [end, string] = processor->tokenizer->postprocess(out_string);
                output_string_ += string;
                callback_(output_string_, !end, {});
                if (!end) { break; }
                chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});
                if (step == 0) static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
            }

            std::cout << std::endl;
        } else {
#endif
            auto model = dynamic_cast<Qwen2VLModel *>(module_.get());
            for (int step = 0; step < 100; step++) {
                model->get_position_ids(input_tensors);
                auto result = (*model)(input_tensors);
                auto outputs = processor->detokenize(result[0]);
                auto out_string = outputs.first;
                auto out_token = outputs.second;
                auto [end, string] = processor->tokenizer->postprocess(out_string);
                output_string_ += string;
                callback_(output_string_, !end, {});
                if (!end) { break; }
                chatPostProcessing(out_token, input_tensors[0], {&input_tensors[1], &input_tensors[2]});
            }
            module_->clear_kvcache();
#ifdef USE_QNN
        }
#endif
    } else if (model_ == Bert) {
        LOGE("Bert model is not supported in this version.");
    } else if (model_ == PhoneLM) {
        // static bool switch_flag = false;
        auto tokenizer = dynamic_pointer_cast<SmolLMTokenizer>(tokenizer_);
        if (chat_template) input_str = tokenizer_->apply_chat_template(input_str);
        if (backend_ == MLLMBackendType::QNN) {
            int chunk_size = 64;
            auto res = tokenizer->tokenizePaddingByChunk(input_str, chunk_size, 49152);
            auto input_tensor = res.second;
            max_new_tokens = tokens_limit - input_tensor.sequence();
            auto real_seq_length = res.first;
            const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
            const int chunk_num = seq_length_padding / chunk_size;
            bool isSwitched = false;

            // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setTotalSequenceLength(real_seq_length);
            // set chunk size for the HeadLinear execute, which can not get the chunk size from Opts
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setChunkSize(chunk_size);

            LlmTextGeneratorOpts opt{
                .max_new_tokens = 1,
                .do_sample = false,
                .is_padding = true,
                .seq_before_padding = real_seq_length,
                .chunk_size = chunk_size,
            };
            std::vector<Tensor> chunked_tensors(chunk_num);
            for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
                chunked_tensors[chunk_id].setBackend(Backend::global_backends[MLLM_CPU].get());
                chunked_tensors[chunk_id].setTtype(INPUT_TENSOR);
                chunked_tensors[chunk_id].reshape(1, 1, chunk_size, 1);
                chunked_tensors[chunk_id].setName("input-chunk-" + to_string(chunk_id));
                chunked_tensors[chunk_id].shallowCopyFrom(&input_tensor, false, {0, 0, chunk_id * chunk_size, 0});

                prefill_module_->generate(chunked_tensors[chunk_id], opt, [&](unsigned int out_token) -> bool {
                    // if (switch_flag && !isSwitched && chunk_id == 0) {
                    if (!isSwitched && chunk_id == 0) {
                        // turn off switching at the first chunk of following inputs
                        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
                        isSwitched = true;
                    }
                    // switch_flag = true;
                    auto out_string = tokenizer_->detokenize({out_token});
                    auto [not_end, output_string] = tokenizer_->postprocess(out_string);
                    if (chunk_id == chunk_num - 1) { // print the output of the last chunk
                        output_string_ += output_string;
                        if (!not_end) {
                            auto profile_res = prefill_module_->profiling("Prefilling");
                            if (profile_res.size() == 3) {
                                profiling_data[0] += profile_res[0];
                                profiling_data[1] = profile_res[1];
                            }
                            callback_(output_string_, !not_end, profiling_data);
                        }
                        callback_(output_string_, !not_end, {});
                    }
                    if (!not_end) { return false; }
                    return true;
                });
                Module::isFirstChunk = false;
            }
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(real_seq_length);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(AUTOREGRESSIVE);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();

            opt = LlmTextGeneratorOpts{
                .max_new_tokens = max_new_tokens - 1,
                .do_sample = false,
                .temperature = 0.3f,
                .top_k = 50,
                .top_p = 0.f,
                .is_padding = false,
            };
            isSwitched = false;
            module_->generate(chunked_tensors.back(), opt, [&](unsigned int out_token) -> bool {
                if (!isSwitched) {
                    static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
                    isSwitched = true;
                }
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data[0] += profile_res[0];
                        profiling_data[2] = profile_res[2];
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(0);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(PROMPT);
            static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
        } else { // CPU
            auto input_tensor = tokenizer_->tokenize(input_str);
            max_new_tokens = tokens_limit - input_tensor.sequence();
            LlmTextGeneratorOpts opt{
                .max_new_tokens = max_new_tokens,
                .do_sample = false,
            };
            module_->generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
                auto out_token_string = tokenizer_->detokenize({out_token});
                auto [not_end, output_string] = tokenizer_->postprocess(out_token_string);
                output_string_ += output_string;
                if (!not_end) {
                    auto profile_res = module_->profiling("Inference");
                    if (profile_res.size() == 3) {
                        profiling_data = profile_res;
                    }
                    callback_(output_string_, !not_end, profiling_data);
                }
                callback_(output_string_, !not_end, {});
                if (!not_end) { return false; }
                return true;
            });
            module_->clear_kvcache();
        }
    }
}
std::vector<float> LibHelper::runForResult(std::string &input_str) {
    LOGE("Running model %d", model_);
    if (model_ == Bert) {
        // auto bert_tokenizer = dynamic_pointer_cast<BertTokenizer>(tokenizer_);
        auto inputs = tokenizer_->tokenizes(input_str);
        auto result = (*module_)(inputs)[0];
        auto output_arr = result.hostPtr<float>();
        return std::vector<float>(output_arr, output_arr + result.count());
    } else {
        return {};
    }
}

LibHelper::~LibHelper() {
    delete processor_;
}
// #endif
