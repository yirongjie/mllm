//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_SplitInput);

// op execute function declarations
template <typename TensorType, typename TensorType1>
GraphStatus splitinputImpl(TensorType &out_0,
                           TensorType &out_1,
                           const TensorType &in_0,
                           const TensorType1 &in_1,
                           const Tensor &num);

// forward declaration of sample cost function
static float splitinputCostFunc(const Op *op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((splitinputImpl<Tensor, Tensor>), "SplitInput")
 */
DEF_PACKAGE_OP((splitinputImpl<Tensor, Tensor>), "SplitInput")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((splitinputImpl<PlainFloatTensor, PlainFloatTensor>), "SplitInput", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((splitinputImpl<PlainFloatTensor, PlainFloatTensor>),
 * "SplitInput", splitinputCostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax: DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core documentations
 */

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax: DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
 * order of parameters listed determines the order of parameters passed into op execution functions
 * if an op does not have a parameter order definition, parameter order passed into Qnn_addNode
 *   will be passed into op execution functions
 * if an op has a parameter order definition, any parameter passed into Qnn_addNode with unlisted
 *   name will be abandoned
 * if two or more op packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANDATORY refers to whether this parameter is required to be provided at Qnn_addNode
 * DEFAULT is used when MANDATORY is false
 *     if provided as Qnn_Param_t*,
 *       DEFAULT will be used for graph construction when this parameter is not provided at
 *       Qnn_addNode
 *     if provided as nullptr,
 *       graph construction will skip this parameter when this parameter is not provided at
 *       Qnn_addNode
 */

/* execute functions for ops */

template <typename TensorType, typename TensorType1>
GraphStatus splitinputImpl(TensorType &out_0,
                           TensorType &out_1,
                           const TensorType &in_0,
                           const TensorType1 &in_1,
                           const Tensor &num) {
    /*
     * add code here
     * */
    /*
     * To have good performance and stability, it is required to avoid heap memory
     * allocation in this function. The heap memory allocation includes but not
     * limited to calling malloc, operator new, constructing STL container objects
     * like std::vector with default allocator, and adding items like calling
     * std::vector::push_back to STL container objects with default allocator.
     *
     * Please check in SDK documentation for more information.
     */

    // default is two.

    size_t o_size = in_1(0, 0, 0, 0);
    size_t x_size = in_1(0, 0, 0, 1);

    auto [b_in, h_in, w_in, d_in] = in_0.dims();

    const size_t dims_0[] = {b_in, o_size, w_in, d_in};
    const size_t dims_1[] = {b_in, x_size, w_in, d_in};

    out_0.set_dims(dims_0);
    out_1.set_dims(dims_1);

    DType dtype = in_0.get_dtype();
    uint32_t bitwidth = 4;

    if (dtype == DType::QUInt8 || dtype == DType::QInt8) {
        bitwidth = 1;

    } else if (dtype == DType::Float16) {
        bitwidth = 2;
    } else if (dtype == DType::Float32) {
        bitwidth = 4;
    }

    const uint8_t *in_ptr = (uint8_t *)in_0.raw_data_const();

    uint8_t *out_ptr_0 = (uint8_t *)out_0.raw_data();
    uint8_t *out_ptr_1 = (uint8_t *)out_1.raw_data();

    memcpy(out_ptr_0, in_ptr, b_in * o_size * w_in * d_in * bitwidth);
    in_ptr += b_in * o_size * w_in * d_in * bitwidth;

    memcpy(out_ptr_1, in_ptr, b_in * x_size * w_in * d_in * bitwidth * 4);

    return GraphStatus::Success;
}

__attribute__((unused)) static float splitinputCostFunc(const Op *op) {
    /*
     * add code here
     * */

    float cost = 0.0; // add cost computation here
    return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_SplitInput);