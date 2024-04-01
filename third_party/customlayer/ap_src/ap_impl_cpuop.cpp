#include <vector>
#include <stdio.h>
#include "bmcpu.h"
#include "cpu_layer.h"
#include "cpu_layer_factory.h"
#include "ap_impl_mycpuop.h"

using namespace std;
namespace bmcpu {

struct ParallelExecutor {
    template<typename Func, typename... ArgTypes>
    static void run(Func functor, const int N, ArgTypes... args){
        //TODO: use std::thread or openmp
        for(int i=0; i<N; i++){
            functor(i, args...);
        }
    }
};

int cpu_mycpuoplayer::get_param(void *param, int param_size) {
    printf("get_param !\n");
    return 0;
}

int cpu_mycpuoplayer::forward(void *raw_param, int param_size) {
    printf("forward !\n");
    return 0;
}

int cpu_mycpuoplayer::shepe_infer(void *param, int param_size,
    const vector<vector<int> > &input_shapes, vector<vector<int> > &output_shapes) {
    printf("shepe_infer !\n");
    return 0;
}

REGISTER_APLAYER_CLASS(AP_CUSTOM, ap_mycpuop);
} /* namespace bmcpu */
