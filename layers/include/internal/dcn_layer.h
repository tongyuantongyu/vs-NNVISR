#pragma once

#include <vector>
#include <string>
#include "NvInfer.h"
#include "config.h"

#define PLUGIN_NAME "DeformConv2d"

namespace nvinfer1 {
class DCNLayerPlugin : public IPluginV2DynamicExt {
 public:
  int getNbOutputs() const noexcept override {
    return 1;
  }

  DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs,
                                int nbInputs, IExprBuilder &exprBuilder) noexcept override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept override;

  // boilerplate
  explicit DCNLayerPlugin(const DCNLayerConfig *config) noexcept;
  DCNLayerPlugin(const DCNLayerPlugin &plugin) noexcept;
  DCNLayerPlugin(const void *data, size_t length) noexcept;
  ~DCNLayerPlugin() noexcept override = default;

  int initialize() noexcept override {
    return 0;
  };

  void terminate() noexcept override {};

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;

  int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void *buffer) const noexcept override;

  const char *getPluginType() const noexcept override {
    return PLUGIN_NAME;
  };

  const char *getPluginVersion() const noexcept override {
    return "1";
  };

  void destroy() noexcept override {};

  IPluginV2DynamicExt *clone() const noexcept override;

  void setPluginNamespace(const char *pluginNamespace) noexcept override {
    this->mPluginNamespace = pluginNamespace;
  };

  const char *getPluginNamespace() const noexcept override {
    return this->mPluginNamespace.c_str();
  };

  DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override {
    return inputTypes[0];
  };

  void attachToContext(
      cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept override {
    cublas_context = cublasContext;
  };

  void detachFromContext() noexcept override {};

  void configurePlugin(
      const DynamicPluginTensorDesc* in, int nbInputs,
      const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

 private:
  template<class F>
  std::pair<DCNLayerInput<F>, DCNLayerOutput<F>> makeView(const void *const *inputs,
                                                          void *const *outputs,
                                                          void *workspace);
  std::string mPluginNamespace;
  DCNLayerConfig config{};
  DCNLayerInternal internal{};

  cublasContext * cublas_context;
};

class DCNLayerPluginCreator : public IPluginCreator {
 public:
  // boilerplate
  DCNLayerPluginCreator() noexcept = default;
  ~DCNLayerPluginCreator() noexcept override = default;

  const char *getPluginName() const noexcept override {
    return PLUGIN_NAME;
  };

  const char *getPluginVersion() const noexcept override {
    return "1";
  };

  const PluginFieldCollection *getFieldNames() noexcept override;

  IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;

  IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;

  void setPluginNamespace(const char *libNamespace) noexcept override {
    mNamespace = libNamespace;
  }

  const char *getPluginNamespace() const noexcept override {
    return mNamespace.c_str();
  }

 private:
  std::string mNamespace;

  const static PluginField mPluginAttributes[];
  const static PluginFieldCollection mPFC;

};

};

#undef PLUGIN_NAME
