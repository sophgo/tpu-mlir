#ifndef AP_TOPK_H
#define AP_TOPK_H
#include "cpu_layer.h"

#define ap_topklayer cpu_topklayer
namespace bmcpu {
class cpu_topklayer : public ap_layer {
public:
    explicit cpu_topklayer() {}
    virtual ~cpu_topklayer() {}
    int forward(void *param, int param_size);

    virtual string get_layer_name () const {
        return "LAYER_NAME";
    }

    int get_param(void *param, int param_size);

    int shepe_infer(void *param, int param_size,
                    const vector<vector<int> > &input_shapes, vector<vector<int> > &output_shapes);

    int dtype_infer(const void *param, size_t param_size, const vector<int> &input_dtypes, vector<int> &output_dtypes) {
        output_dtypes = {CPU_DTYPE_FP32, CPU_DTYPE_FP32};
        return 0;
    }
private:
    int K_ = 0;
    int axis_ = 0;
};

} /* namespace bmcpu */
#endif // AP_TOPK_H
