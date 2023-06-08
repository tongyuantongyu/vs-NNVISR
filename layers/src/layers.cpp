#include "helper.h"
#include "dcn_layer.h"

namespace UDOLayers {

template<typename T>
struct PluginRegistrar {
#ifdef AUTO_REGISTER_PLUGIN
  PluginRegistrar() noexcept
  {
    getPluginRegistry()->registerCreator(instance, "");
  }
#endif

  T instance{};
};

namespace {
PluginRegistrar<nvinfer1::DCNLayerPluginCreator> _mDCNLayerPluginCreator{};
}

bool registerDCNLayerPlugin() {
  return getPluginRegistry()->registerCreator(_mDCNLayerPluginCreator.instance, "");
}

bool registerPlugins() {
  return registerDCNLayerPlugin();
}

}
