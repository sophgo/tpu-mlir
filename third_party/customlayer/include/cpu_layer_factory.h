#ifndef CPU_LAYER_FACTORY_H_
#define CPU_LAYER_FACTORY_H_

#include "cpu_layer.h"

namespace bmcpu {

class cpu_layer;

class CpuLayerRegistry {
public:
    typedef std::shared_ptr<cpu_layer> (*Creator)();
    typedef std::map<int, Creator> CreatorLayerRegistry;

    /* Create Layer Factor */
    static CreatorLayerRegistry& LayerRegistry() {
        static CreatorLayerRegistry g_registry_;
        return g_registry_;
    }

    /* Reigster Layer Creator to Factory */
    static void addCreator(const int layer_type, Creator Creator) {
        auto& layer_registry = LayerRegistry();

        if (layer_registry.count(layer_type)==0){
            //cout << "register layer_type: " << layer_type << endl;
            layer_registry[layer_type] = Creator;
        } else {
            cout << "[WARNING] layer_type:" << layer_type << " already registered on cpu" << endl;
        }
    }

    /* unified layer create interface : search layer factor and find corresponding layer creator */
    static std::shared_ptr<cpu_layer> createlayer(const int layer_type) {
        auto& layer_registry = LayerRegistry();
        if (layer_registry.count(layer_type) != 0) {
            return layer_registry[layer_type]();
        }
        printf("Cannot create bmcpu layer for layer_type:%d\n", layer_type);
        exit(-1);
    }

private:
    CpuLayerRegistry() {}

};


class Cpulayerregisterer {
public:
    Cpulayerregisterer(const int& layer_type,
                       std::shared_ptr<cpu_layer> (*Creator)()) {
        CpuLayerRegistry::addCreator(layer_type, Creator);
    }
};


/* create layer object like cpu_softmaxlayer, and register */
#define REGISTER_APLAYER_CLASS(layer_type, layer_name)           \
  std::shared_ptr<cpu_layer> Creator_##layer_name##Layer() {      \
    return std::shared_ptr<cpu_layer>(new layer_name##layer());   \
  }                                                               \
  static Cpulayerregisterer g_Creator_f_##type(layer_type, Creator_##layer_name##Layer);


} /* namespace bmcpu */

#endif /* CPU_LAYER_FACTORY_H_ */
