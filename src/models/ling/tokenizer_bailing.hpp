#ifndef TOKENIZATION_BAILING_LITE_HPP
#define TOKENIZATION_BAILING_LITE_HPP

#include "tokenizers/BPE/Bpe.hpp"
#include <string>
#include <vector>

using namespace mllm;

/**
 * @class BaiLingTokenizer
 * @brief BaiLing模型的Tokenizer，为模型的特殊token提供了定制化处理。
 */
class BaiLingTokenizer final : public BPETokenizer {
public:
    /**
     * @brief BaiLingTokenizer的构造函数。
     * @param vocab_file .mllm词汇文件的路径。
     * @param merge_file .merges.txt 合并规则文件的路径。
     */
    explicit BaiLingTokenizer(const std::string &vocab_file, const std::string &merge_file) :
        BPETokenizer(vocab_file) {
        auto merge_file_stream = std::ifstream(merge_file);
        if (!merge_file_stream.good()) {
            std::cout << "merge file is broken\n";
            exit(0);
        }
        std::string line;
        unsigned rank = 0;
        std::unordered_map<std::string, unsigned int> bpe_ranks_;
        while (std::getline(merge_file_stream, line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] == '#') {
                continue;
            }
            bpe_ranks_[line] = rank;
            rank++;
        }
        BPETokenizer::setMergeRank(bpe_ranks_);
        chat_template_pre = "<role>SYSTEM</role>You are Ling, an assistant created by inclusionAI<role>HUMAN</role>";
        chat_template_end = "<role>ASSISTANT</role>";

        special_tokens_ = {
            bos_token_string_, eos_token_string_, "[CLS]", "[gMASK]",
            "<role>", "</role>",
            "<|arithmetic_start|>", "<|arithmetic_end|>",
            "<|number_start|>", "<|number_end|>"
        };
        for (int i = 0; i <= 100; ++i) {
            special_tokens_.push_back("<|reserved_token_" + std::to_string(i) + "|>");
        }
        special_tokens_.push_back("<role>SYSTEM</role>");
        special_tokens_.push_back("<role>HUMAN</role>");
        special_tokens_.push_back("<role>BOT</role>");
    }

    /**
     * @brief 重写的tokenize方法，用于处理包含特殊token的文本。
     * @param text 输入的待分词字符串。
     * @param tokens 输出的token ID向量。
     * @param bos 是否在开头添加BOS (Beginning of Sequence) token。
     */
    Tensor tokenize(const std::string &text, string name = "input_ids", BackendType type = MLLM_CPU) override {

        //replace "Ġ" (U+0120) with ' '
        std::string text1 = text;
        const std::string target = " ";
        size_t pos = 0;
        while ((pos = text1.find(target, pos)) != std::string::npos) {
            text1.replace(pos, target.length(), u8"Ġ");
            pos += 1;
        }


        std::vector<token_id_t> tokens_id;
        
        auto parts = _splitWithDelimiters(text1, special_tokens_);

        for (const auto& part : parts) {
            if (part.empty()) continue;

            auto it = vocab_map_.find(part);
            if (it != vocab_map_.end()) {
                tokens_id.push_back(it->second);
            } else {
                std::vector<token_id_t> sub_tokens;
                BPETokenizer::tokenize(part, sub_tokens, false); // bos=false
                tokens_id.insert(tokens_id.end(), sub_tokens.begin(), sub_tokens.end()-1);
            }
        }
        
        return Tokenizer::tokens2Input(tokens_id, name, type);
    }


    /**
     * @brief [新增] 将一个token ID的向量解码成字符串。
     * @param tokens 待解码的token ID向量。
     * @return 解码后的字符串。
     */
    std::string detokenize(const std::vector<token_id_t> &tokens) override {
        // 对于此模型，直接调用基类的实现即可满足要求。
        // 基类实现会遍历ID，查找对应的token字符串并拼接起来。
        return BPETokenizer::detokenize(tokens);
    }

    /**
     * @brief [保留] 从模型的输出Tensor中解码出单个token，用于流式生成。
     */
    std::pair<std::string, unsigned> detokenize(Tensor &result) override {
        assert(result.batch() == 1);
        assert(result.head() == 1);
        vector<float> scores;
        for (int i = 0; i < result.dimension(); ++i) {
            auto value = result.dataAt<float>(0, 0, result.sequence() - 1, i);
            scores.push_back(value);
        }
        auto token_idx = this->argmax(scores);
        return {BPETokenizer::detokenize({token_idx}), token_idx};
    }

    /**
     * @brief 对解码出的文本进行后处理。
     * @param text 解码出的单个token字符串。
     * @return 一个pair，第一个bool值表示是否继续生成，第二个string是最终要显示的文本。
     */
    std::pair<bool, std::string> postprocess(std::string &text) override {
        const std::string target = u8"Ġ";
        size_t pos = 0;
        while ((pos = text.find(target, pos)) != std::string::npos) {
            text.replace(pos, target.length(), " ");
            pos += 1;
        }
        if (text == this->eos_token_string_||
            u8"Ċ" == text) {
            return {false, ""};
        }

        if (text == this->bos_token_string_ ||
            text == "<role>" || text == "</role>" ||
            text.rfind("<|reserved_token_", 0) == 0 ||
            text.rfind("<role>", 0) == 0)
        {
            return {true, ""};
        }

        return {true, text};
    }

private:
    const std::string bos_token_string_ = "<|startoftext|>";
    const std::string eos_token_string_ = "<|endoftext|>";

    std::vector<std::string> special_tokens_;

    std::vector<std::string> _splitWithDelimiters(const std::string &str, const std::vector<std::string> &delimiters) const {
        std::vector<std::string> result;
        size_t last = 0;

        while (last < str.size()) {
            size_t min_pos = std::string::npos;
            std::string best_delim = "";

            for (const auto& delim : delimiters) {
                if (!delim.empty()) {
                    size_t found_pos = str.find(delim, last);
                    if (found_pos != std::string::npos && (min_pos == std::string::npos || found_pos < min_pos)) {
                        min_pos = found_pos;
                        best_delim = delim;
                    }
                }
            }

            if (min_pos != std::string::npos) {
                if (min_pos > last) {
                    result.push_back(str.substr(last, min_pos - last));
                }
                result.push_back(best_delim);
                last = min_pos + best_delim.length();
            } else {
                result.push_back(str.substr(last));
                break;
            }
        }
        return result;
    }
};

#endif // TOKENIZATION_BAILING_LITE_HPP