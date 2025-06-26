/**
 * @file demo_bailing_moe.cpp
 * @brief A demo for using Bailing MoE model.
 * @author Rongjie Yi
 * @date 2025-07-01
 *
 */
#include "cmdline.h"
#include "models/ling/configuration_bailing_moe.hpp"
#include "models/ling/modeling_bailing_moe.hpp"
#include "models/ling/tokenizer_bailing.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    std::iostream::sync_with_stdio(false);

    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/ling_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/ling_merges.txt");
#ifdef ARM
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/ling-lite-base-1.5-kai_q4_0.mllm");
#else
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/ling-lite-base-1.5-q4_0.mllm");
#endif
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = BaiLingTokenizer(vocab_path, merge_path);
    BailingMoeConfig config(tokens_limit);
    auto model = BailingMoeForCausalLM(config);
    model.load(model_path);

    vector<string> in_strs = {
        // "Who are you?",
        " Give me a short introduction to large language model.",
    };
    for (int i = 0; i < in_strs.size(); ++i) {
        auto input_str = tokenizer.apply_chat_template(in_strs[i]);
        auto input_tensor = tokenizer.tokenize(input_str);
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;

        LlmTextGeneratorOpts opt{
            .max_new_tokens = 120,
            .do_sample = false,
            .temperature = 0.3F,
            .top_k = 50,
            .top_p = 0.F,
        };
        model.generate(input_tensor, opt, [&](unsigned int out_token) -> bool {
            auto out_string = tokenizer.detokenize({out_token});
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { return false; }
            std::cout << output_string << std::flush;
            return true;
        });
        std::cout << "\n";
        model.clear_kvcache();
        model.profiling();
    }
}
