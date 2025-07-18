#include <iostream>
#include "cmdline.h"
#include "models/qwen2_vl/configuration_qwen2_vl.hpp"
#include "models/qwen2_vl/vtp/modeling_qwen2_vl.hpp"
#include "models/qwen2_vl/vtp/processing_qwen2_vl.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/qwen2vl_vocab.mllm");
    cmdParser.add<string>("merge", 'e', "specify mllm merge file path", false, "../vocab/qwen2vl_merges.txt");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/qwen-2-vl-2b-instruct-q4_k.mllm");
    cmdParser.add<string>("billion", 'b', "[2B | 7B |]", false, "2B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 800);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.add("premerge", 'p', "enable pre-ViT image token merging");
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merge_path = cmdParser.get<string>("merge");
    string model_path = cmdParser.get<string>("model");
    string model_billion = cmdParser.get<string>("billion") == "2B" ? "1.5b" : cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");
    use_pre_vit_merge = cmdParser.exist("premerge");

    ParamLoader param_loader(model_path);
    auto processor = Qwen2VLProcessor(vocab_path, merge_path);
    Qwen2VLConfig config(tokens_limit, model_billion);
    auto model = Qwen2VLModel(config);
    model.load(model_path);

    vector<string> in_imgs = {
        // "../assets/bus.png",
        "../assets/two_cats.jpg",
        // "../assets/bird_image.jpg",
    };
    vector<string> in_strs = {
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image.",
    };

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str = in_strs[i];
        in_str = processor.tokenizer->apply_chat_template(in_str);
        auto input_tensor = processor.process(in_str, in_imgs[i]);
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        for (int step = 0; step < 100; step++) {
            model.get_position_ids(input_tensor);
            auto result = model(input_tensor);
            auto outputs = processor.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor[0], {&input_tensor[1], &input_tensor[2]});
        }
        printf("\n");
        model.clear_kvcache();
        model.profiling();
    }

    return 0;
}