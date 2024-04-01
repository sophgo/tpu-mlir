#ifndef _CPU_LAYER_H_
#define _CPU_LAYER_H_

#include <memory>
#include "bmcpu_macro.h"
#include "customap_common.h"
#include "cpu_layer_factory.h"

#define bmap bmcpu
#define ap_layer cpu_layer
#define MAX_SHAPE_DIMS 8
namespace bmcpu {

inline int bmcpu_shape_count(vector<int> shape_v)
{
    int count = 1;
    for(auto& shape : shape_v)
        count *= shape;

    return count;
}

typedef union {
    int int_t;
    float float_t;
    // max size of int and float array is set as 16
    int int_arr_t[16];
    float float_arr_t[16];
} custom_param_t;

class cpu_layer {
public:
    cpu_layer() {}
    ~cpu_layer() {}

    /* dowork */
    virtual int forward(void *parm, int param_size);
    int mlir_inference(void *param, int param_size,
                       const vector<float *> &input_tensors,
                       const vector<float *> &output_tensors,
                       const vector<vector<int>> &input_shapes,
                       vector<vector<int>> &output_shape);
    int set_common_param(const vector<float *> &input_tensors,
                         const vector<vector<int>> &input_shapes,
                         const vector<float *> &output_tensors,
                         vector<vector<int>> &output_shape);

    virtual int get_param(void *param, int param_size) = 0;
    virtual int shepe_infer(void *param, int param_size,
                            const vector<vector<int>> &input_shapes,
                            vector<vector<int>> &output_shapes);

    virtual int dtype_infer(const void *param, size_t param_size, const vector<int> &input_dtypes,
                      vector<int> &output_dtypes) {
        std::cout << "Output dtypes cannot be obtained." << std::endl;
        return -1;
    }

protected:

    vector<float *> input_tensors_;
    vector<vector<int>> input_shapes_;

    vector<float *> output_tensors_;  /* TODO: int8 */
    vector<vector<int>>* output_shapes_;

    /* layer specific param */
    void* layer_param_;
};


inline int cpu_layer::set_common_param(
           const vector<float *>& input_tensors,
           const vector<vector<int>>& input_shapes,
           const vector<float *>& output_tensors,
           vector<vector<int>>& output_shapes)
{
    CPU_ASSERT(input_tensors.size() == input_shapes.size());
    CPU_ASSERT(output_tensors.size() == output_shapes.size());

    input_tensors_    = input_tensors;
    input_shapes_     = input_shapes;

    output_tensors_   = output_tensors;
    output_shapes_    = &output_shapes;

    return 0;
}

inline int cpu_layer::forward(void *param, int param_size) {
    std::cout << "cpu_layer forward" << std::endl;
    return 0;
}

inline int cpu_layer::shepe_infer(void *param, int param_size,
                                  const vector<vector<int>> &input_shapes,
                                  vector<vector<int>> &output_shapes) {
    std::cout << "cpu_layer shepe_infer" << std::endl;
    return 0;
}

inline int cpu_layer::mlir_inference(void *param, int param_size,
                                     const vector<float *> &input_tensors,
                                     const vector<float *> &output_tensors,
                                     const vector<vector<int>> &input_shapes,
                                     vector<vector<int>> &output_shape) {
    this->get_param(param, param_size);
    this->set_common_param(input_tensors, input_shapes, output_tensors,
                            output_shape);
    this->forward(param, param_size);
    return 0;
}

} /* namespace bmcpu */

extern "C" {
    bmcpu::cpu_layer* createLayerInstance(const char* layerType);
}

#endif /* _CPU_LAYER_H_ */
