
#ifndef MLLM_TYPES_H
#define MLLM_TYPES_H
#include "OpDefined.hpp"
#include "DataType.hpp"
#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>
#include <cassert>
#include <cstdint>
#include <Log.h>
using std::string;
using std::vector;
using std::map;

typedef map<std::string, float> OpParam;

// #define DEBUGSAVETENSOR
// #define DEBUGOPTIME

#define LLAMAFILE_SGEMM
inline int KVCache_TYPE = 16;
inline int KVCacheSageDtypeBit = 8; // 8 or 16
inline int KVCache_batch = 1;
typedef enum {
    MLLM_CPU,
    MLLM_OPENCL,
    MLLM_QNN,
    MLLM_XNNPACK,
} BackendType;

enum TensorStatus {
    // TENSOR_DYNAMIC,
    TENSOR_STATIC_INIT,
    TENSOR_STATIC_READY,
    TENSOR_STATIC_TRACE,
    TENSOR_UNDEFINED,
};

enum CallableType {
    OP,
    TENSOR_FUNC
};

enum ErrorCode {
    MLLM_NO_ERROR = 0,
    OUT_OF_MEMORY = 1,
    NOT_SUPPORT = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION = 4,
    INVALID_VALUE = 5,
};

enum DataType {
    MLLM_TYPE_F32 = 0,
    MLLM_TYPE_F16 = 1,
    MLLM_TYPE_Q4_0 = 2,
    MLLM_TYPE_Q4_1 = 3,
    MLLM_TYPE_Q8_0 = 8,
    MLLM_TYPE_Q8_1 = 9,
    MLLM_TYPE_Q8_PER_TENSOR = 10,
    // k-quantizations
    MLLM_TYPE_Q4_K = 12,
    MLLM_TYPE_Q6_K = 14,
    MLLM_TYPE_Q8_K = 15,
    MLLM_TYPE_I8,
    MLLM_TYPE_I16,
    MLLM_TYPE_I32,
    MLLM_TYPE_Q4_0_4_4 = 19,
    MLLM_TYPE_Q4_0_4_8 = 20,
    MLLM_TYPE_Q4_0_8_8 = 21,
    MLLM_TYPE_Q8_0_4_4 = 22,
    // 2-bit quantizations
    MLLM_TYPE_Q3_K = 23, //
    MLLM_TYPE_Q2_K = 24,
    MLLM_TYPE_Q1_K = 25,    //
    MLLM_TYPE_IQ2_XXS = 26, //
    MLLM_TYPE_IQ2_XS = 27,  //
    MLLM_TYPE_IQ1_S = 28,   //
    MLLM_TYPE_IQ1_M = 29,   //
    MLLM_TYPE_IQ2_S = 30,

    MLLM_TYPE_KLEIDIAI_Q4_0 = 31,
    MLLM_TYPE_Q8_0F = 32, // quantized with float scale
    MLLM_TYPE_Q2_0 = 33,  // 2-bits quantization

    MLLM_TYPE_COUNT,
};

enum ChlType {
    BSHD = 0,
    BHSD,
    BHDS = 2,

    BCTHW = 3,
    BTHWC = 4,
    BWCTH = 5,

    SBHD = 10,
    BDHS = 11,
    BDSH = 12,
    DBHS = 13
};

inline std::map<std::vector<int>, ChlType> Chls2Type = {
    {{0, 2, 3, 1}, BDHS},
    {{0, 1, 3, 2}, BHDS},
    {{0, 2, 1, 3}, BSHD},
    {{0, 1, 2, 3}, BHSD},
    {{1, 2, 0, 3}, SBHD},
    {{0, 3, 2, 1}, BDSH},
    {{1, 2, 3, 0}, DBHS},
    {{0, 1, 2, 3, 4}, BTHWC},
    {{0, 2, 3, 4, 1}, BCTHW},
    {{0, 3, 4, 1, 2}, BWCTH}};

enum TensorType {
    INPUT_TENSOR = 0, // used for input of the model
    NORMAL_TENSOR,
    GRAPH_OUTPUT, // used for output of a graph
    OUTPUT_TENSOR,
};

enum Chl {
    BATCH = 0,
    SEQUENCE = 1,
    HEAD = 2,
    DIMENSION = 3,

    HD = 113,   // only use for split attn.in_proj
    D_HD = 313, // only use for split attn.in_proj
    D_DH = 331, // only use for split attn.in_proj

    CHANNLE = 1,
    TIME = 2,
    HEIGHT = 3,
    WIDTH = 4,

    THW = 234,

};

enum AttnQKVSplitType {
    SPLIT_NONE = 0,
    SPLIT_HD = Chl::HD,
    SPLIT_D_HD = Chl::D_HD,
};

enum AttnPostQkvNormType {
    PostQkv_NONE = 0,
    PostQkv_LayerNorm,
    PostQkv_RMSNorm,
};

#define ANYDIM -198098

enum PaddingType {
    SAME,
    VALID
};

enum RoPEType {
    NONE = 0,
    LLAMAROPE = 2,
    PERSIMMONROPE = 3,
    HFHUBROPE = 4,
    MLAROPE = 5,
    NTKROPE = 6,
};

enum RoPEThetaType {
    DEFAULT = 0,
    LLAMA3 = 1,
};

enum ExecutionType {
    PROMPT = 0,
    AUTOREGRESSIVE = 1,
};

static string DataTypeName(DataType dataType) {
    switch (dataType) {
    case MLLM_TYPE_F32:
        return "F32";
    case MLLM_TYPE_F16:
        return "F16";
    case MLLM_TYPE_I32:
        return "I32";
    case MLLM_TYPE_I16:
        return "I16";
    case MLLM_TYPE_I8:
        return "I8";
    case MLLM_TYPE_Q8_PER_TENSOR:
        return "Q8_PER_TENSOR";
    case MLLM_TYPE_Q4_0:
        return "Q4_0";
    case MLLM_TYPE_Q4_K:
        return "Q4_K";
    case MLLM_TYPE_Q6_K:
        return "Q6_K";
    case MLLM_TYPE_Q8_0:
        return "Q8_0";
    case MLLM_TYPE_Q8_K:
        return "Q8_K";
    case MLLM_TYPE_Q4_1:
        return "Q4_1";
    case MLLM_TYPE_Q8_1:
        return "Q8_1";
    case MLLM_TYPE_Q4_0_4_4:
        return "Q4_0_4_4";
    case MLLM_TYPE_Q4_0_4_8:
        return "Q4_0_4_8";
    case MLLM_TYPE_Q4_0_8_8:
        return "Q4_0_8_8";
    case MLLM_TYPE_Q8_0_4_4:
        return "Q8_0_4_4";
    case MLLM_TYPE_Q3_K:
        return "Q3_K";
    case MLLM_TYPE_Q2_K:
        return "Q2_K";
    case MLLM_TYPE_Q1_K:
        return "Q1_K";
    case MLLM_TYPE_IQ2_XXS:
        return "IQ2_XXS";
    case MLLM_TYPE_IQ2_XS:
        return "IQ2_XS";
    case MLLM_TYPE_IQ1_S:
        return "IQ1_S";
    case MLLM_TYPE_IQ1_M:
        return "IQ1_M";
    case MLLM_TYPE_IQ2_S:
        return "IQ2_S";
    case MLLM_TYPE_KLEIDIAI_Q4_0:
        return "KLEIDIAI_Q4_0";
    case MLLM_TYPE_Q8_0F:
        return "Q8_0F";
    case MLLM_TYPE_Q2_0:
        return "Q2_0";
    case MLLM_TYPE_COUNT:
        return "COUNT";
    default:
        return "Unknown";
    }
}

static size_t DataTypeSize(DataType dtype, uint64_t count = 1) {
    switch (dtype) {
    case MLLM_TYPE_F32:
        return sizeof(float) * count;
    case MLLM_TYPE_F16:
        return sizeof(mllm_fp16_t) * count;
    case MLLM_TYPE_I32:
        return sizeof(int) * count;
    case MLLM_TYPE_I16:
        return sizeof(short) * count;
    case MLLM_TYPE_I8:
        return sizeof(char) * count;
    case MLLM_TYPE_Q4_0:
        return (sizeof(block_q4_0)) * count / (QK4_0);
    case MLLM_TYPE_Q4_K:
        return (sizeof(block_q4_K)) * count / (QK_K);
    case MLLM_TYPE_Q6_K:
        return (sizeof(block_q6_K)) * count / (QK_K);
    case MLLM_TYPE_Q8_PER_TENSOR:
        return sizeof(char) * count;
    case MLLM_TYPE_Q8_0:
        return (sizeof(block_q8_0)) * count / (QK8_0);
    case MLLM_TYPE_Q8_K:
        return (sizeof(block_q8_K)) * count / (QK_K);
    case MLLM_TYPE_Q4_1:
    case MLLM_TYPE_Q8_1:
        return -1;
    case MLLM_TYPE_Q4_0_4_4:
        return (sizeof(block_q4_0x4)) * count / (QK4_0 * 4);
    case MLLM_TYPE_Q4_0_4_8:
        return (sizeof(block_q4_0x8)) * count / (QK4_0 * 8);
    case MLLM_TYPE_Q4_0_8_8:
        return (sizeof(block_q4_0x8)) * count / (QK4_0 * 8);
    case MLLM_TYPE_Q8_0_4_4:
        return (sizeof(block_q8_0x4)) * count / (QK8_0 * 4);
    case MLLM_TYPE_Q3_K:
        return (sizeof(block_q3_K)) * count / (QK_K);
    case MLLM_TYPE_Q2_K:
        return (sizeof(block_q2_K)) * count / (QK_K);
    case MLLM_TYPE_Q1_K:
        return -1;
    case MLLM_TYPE_IQ2_XXS:
        return (sizeof(block_iq2_xxs)) * count / (QK_K);
    case MLLM_TYPE_IQ2_XS:
        return -1;
    case MLLM_TYPE_IQ1_S:
        return -1;
    case MLLM_TYPE_IQ1_M:
        return -1;
    case MLLM_TYPE_IQ2_S:
        return -1;
    case MLLM_TYPE_KLEIDIAI_Q4_0:
        return sizeof(uint8_t) * count;
    case MLLM_TYPE_Q8_0F:
        return (sizeof(block_q8_0f)) * count / (QK8_0F);
    case MLLM_TYPE_Q2_0:
        return (sizeof(block_q2_0)) * count / (QK2_0);
    case MLLM_TYPE_COUNT:
        return 0;
    default:
        return 0;
    }
}
#ifdef __cplusplus
namespace mllm {
// TODO: copy from MNN; need to recode #UNUSED
struct BackendConfig {
    enum MemoryMode {
        Memory_Normal = 0,
        Memory_High,
        Memory_Low
    };

    MemoryMode memory = Memory_Normal;

    enum PowerMode {
        Power_Normal = 0,
        Power_High,
        Power_Low
    };

    PowerMode power = Power_Normal;

    enum PrecisionMode {
        Precision_Normal = 0,
        Precision_High,
        Precision_Low
    };

    PrecisionMode precision = Precision_Normal;

    /** user defined context */
    void *sharedContext = nullptr;
};

} // namespace mllm
#endif //__cplusplus
#endif // MLLM_TYPES_H
