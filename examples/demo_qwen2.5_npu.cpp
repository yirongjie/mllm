#include "QNNBackend.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include "cmdline.h"
#include "models/qwen/configuration_qwen.hpp"
#include "models/qwen/modeling_qwen_npu.hpp"
#include "models/qwen/modeling_qwen.hpp"
#include "models/qwen/tokenization_qwen.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2.5_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2.5_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen-2-int8-int32bias-0kproj-test.mllm");
    cmdParser.add<string>("billion", 'b', "[0.5B | 1.8B | 1.5B]", false, "1.5B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = QWenTokenizer(vocab_path, merge_path);
    QWenConfig config(tokens_limit, model_billion, RoPEType::HFHUBROPE);
    auto model = v2::QWenForCausalLM_NPU(config, 32);

    Module::initBackend(MLLM_QNN);

    model.load(model_path);
    auto decoding_model = QWenForCausalLM(config);
    decoding_model.load("../models/qwen-2.5-1.5b-instruct-q4_k.mllm");

    vector<string> in_strs = {
        " Give me a short introduction to large language model.",
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto [real_seq_length, input_tensor] = tokenizer.tokenizeWithPadding(input_str, 32, config.vocab_size);
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        // set total seq length for HeadLinear execute, which can not get the real seq length from Opts
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setTotalSequenceLength(real_seq_length);

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 1,
            .do_sample = false,

            .is_padding = true,
            .seq_before_padding = real_seq_length,
        };
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            std::cout << output_string << std::flush;
            return true;
        });

        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(real_seq_length);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(AUTOREGRESSIVE);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();

        LlmTextGeneratorOpts decoding_opt{
            .max_new_tokens = 50,
            .do_sample = false,
            .temperature = 0.3f,
            .top_k = 50,
            .top_p = 0.f,
            .is_padding = false,
        };
        bool isSwitched = false;
        decoding_model.generate(input_tensor, decoding_opt, [&](unsigned int out_token) -> bool {
            // call only once of switchDecodeTag
            if (!isSwitched) {
                static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();

                isSwitched = true;
            }
            auto out_string = tokenizer.detokenize({out_token});
            auto [isOk, print_string] = tokenizer.postprocess(out_string);
            if (isOk) {
                std::cout << print_string << std::flush;
            } else {
                return false;
            }
            return true;
        });

        // turn on switching, set sequence length and execution type
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setCurSequenceLength(0);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->setExecutionType(PROMPT);
        static_cast<CPUBackend *>(Backend::global_backends[MLLM_CPU].get())->toggleSwitching();
        std::cout << "\n";

        if (!std::filesystem::exists("qnn_context.bin")) {
            static_cast<QNNBackend *>(Backend::global_backends[MLLM_QNN].get())->saveQNNContext();
        }
    }
}
