#include <cstring>
#include <exception>
#include <cassert>
#include <cuda_fp16.h>
#include "helper.h"
#include "config.h"
#include "dcn_layer.h"
#include "dcn_layer_impl.h"

namespace nvinfer1 {

constexpr static struct {
  uint16_t hdr;
  int padding_test;
  float float_test;
} signature = {0xfeff, 0x1, 1.0f};

DCNLayerPlugin::DCNLayerPlugin(const DCNLayerConfig *config) noexcept {
  this->config = *config;
}

DCNLayerPlugin::DCNLayerPlugin(const DCNLayerPlugin &plugin) noexcept {
  this->config = plugin.config;
  this->mPluginNamespace = plugin.mPluginNamespace;
}

DCNLayerPlugin::DCNLayerPlugin(const void *data, size_t length) noexcept {
  assert(length == sizeof(config));
  std::memcpy(&config, (const char *) (data), sizeof(config));
}

size_t DCNLayerPlugin::getSerializationSize() const noexcept {
  return sizeof(signature) + sizeof(config);
}

void DCNLayerPlugin::serialize(void *buffer) const noexcept {
  std::memcpy(buffer, &signature, sizeof(signature));
  std::memcpy((char *) (buffer) + sizeof(signature), &config, sizeof(config));
}

IPluginV2DynamicExt *DCNLayerPlugin::clone() const noexcept {
  auto p = new DCNLayerPlugin(&config);
  p->setPluginNamespace(this->getPluginNamespace());
  return p;
}

enum {
  Input,
  Offset,
  Mask,
  Weight,
  Bias,
  nInput
};

void DCNLayerPlugin::configurePlugin(
    const DynamicPluginTensorDesc *in, int nbInputs,
    const DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  assert(nbInputs == nInput);
  assert(nbOutputs == 1);

  shape_t<4> offset, mask;

  internal.data_type = int32_t(in->desc.type);
  internal.input.gather_from(in[Input].desc.dims.d, 0, 1, 2, 3);
  offset.gather_from(in[Offset].desc.dims.d, 0, 1, 2, 3);
  mask.gather_from(in[Mask].desc.dims.d, 0, 1, 2, 3);
  internal.weight.gather_from(in[Weight].desc.dims.d, 0, 1, 2, 3);
  internal.bias[0] = in[Bias].desc.dims.d[0];

  if (internal.input[0] != -1 &&
      internal.input[1] != -1 &&
      internal.input[2] != -1 &&
      internal.input[3] != -1) {
    // check and calc only when we know exact dimension.

    const auto [n, cin, h, w] = internal.input;

    // weight: input channel match
    assert(cin == internal.weight[1]);

    // offset: stack at dim 1 of offset_h, offset_w
    assert(n == offset[0]);
    const auto [cout, _, kh, kw] = internal.weight;
    const offset_t deformable_channels = kh * kw * config.deformable_groups;
    assert(deformable_channels * 2 == offset[1]);
    const offset_t oh = (h + 2 * config.padding.h - config.dilation.h * (kh - 1) - 1) / config.stride.h + 1;
    const offset_t ow = (w + 2 * config.padding.w - config.dilation.w * (kw - 1) - 1) / config.stride.w + 1;
    assert(oh == offset[2]);
    assert(ow == offset[3]);
    internal.offset = {n, config.deformable_groups, kh, kw, 2, offset[2], offset[3]};

    // mask
    assert(n == mask[0]);
    assert(deformable_channels == mask[1]);
    assert(oh == mask[2]);
    assert(ow == mask[3]);
    internal.mask = {n, config.deformable_groups, kh, kw, mask[2], mask[3]};

    // bias: output channel match
    assert(internal.weight[0] == internal.bias[0]);

    internal.output = {n, cout, offset[2], offset[3]};
    internal.im2col_buffer = {n, cin, kh, kw, offset[2], offset[3]};
  }

}

DimsExprs DCNLayerPlugin::getOutputDimensions(int outputIndex, const DimsExprs *inputs,
                                              int nbInputs, IExprBuilder &exprBuilder) noexcept {

  switch (outputIndex) {
    case 0: {
      const auto n = inputs[Input].d[0];
      const auto h = inputs[Input].d[2];
      const auto w = inputs[Input].d[3];
      const auto c = inputs[Weight].d[0];
      const auto kh = inputs[Weight].d[2];
      const auto kw = inputs[Weight].d[3];

      using op = DimensionOperation;

      // (h - config.dilation.h * (kh - 1) + 2 * config.padding.h - 1) / config.stride.h + 1
      auto oh = exprBuilder.operation(op::kSUB, *kh, *exprBuilder.constant(1));
      oh = exprBuilder.operation(op::kPROD, *oh, *exprBuilder.constant(config.dilation.h));
      oh = exprBuilder.operation(op::kSUB, *h, *oh);
      oh = exprBuilder.operation(op::kSUM, *oh, *exprBuilder.constant(2 * config.padding.h - 1));
      oh = exprBuilder.operation(op::kFLOOR_DIV, *oh, *exprBuilder.constant(config.stride.h));
      oh = exprBuilder.operation(op::kSUM, *oh, *exprBuilder.constant(1));

      auto ow = exprBuilder.operation(op::kSUB, *kw, *exprBuilder.constant(1));
      ow = exprBuilder.operation(op::kPROD, *ow, *exprBuilder.constant(config.dilation.w));
      ow = exprBuilder.operation(op::kSUB, *w, *ow);
      ow = exprBuilder.operation(op::kSUM, *ow, *exprBuilder.constant(2 * config.padding.w - 1));
      ow = exprBuilder.operation(op::kFLOOR_DIV, *ow, *exprBuilder.constant(config.stride.w));
      ow = exprBuilder.operation(op::kSUM, *ow, *exprBuilder.constant(1));
      return DimsExprs{4, {
          n,  // batch size
          c,  // out channel
          oh, // out height
          ow  // out width
      }};
    }

    default:
      // output count exceed.
      assert(false);
      PLUGIN_UNREACHABLE;
  }
};

bool DCNLayerPlugin::supportsFormatCombination(int pos,
                                               const PluginTensorDesc *inOut,
                                               int nbInputs,
                                               int nbOutputs) noexcept {
  if (inOut[pos].format != TensorFormat::kLINEAR) {
    return false;
  }

  switch (pos) {
    case Input:return inOut[Input].type == DataType::kFLOAT || inOut[Input].type == DataType::kHALF;

    case Offset:
    case Mask:
    case Weight:
    case Bias:
    case Bias + 1:
      return inOut[pos].type == inOut[Input].type;

    default:
      // inOut count exceed.
      assert(false);
      PLUGIN_UNREACHABLE;
  }
}

size_t DCNLayerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                        int nbInputs,
                                        const nvinfer1::PluginTensorDesc *outputs,
                                        int nbOutputs) const noexcept {
  shape_t <6> im2col {
      inputs[Input].dims.d[0],
      inputs[Input].dims.d[1],
      inputs[Weight].dims.d[2],
      inputs[Weight].dims.d[3],
      inputs[Offset].dims.d[2],
      inputs[Offset].dims.d[3]
  };
  return im2col.count() * (inputs->type == DataType::kHALF ? 2 : 4);
}

template<class F>
std::pair<DCNLayerInput<F>, DCNLayerOutput<F>> DCNLayerPlugin::makeView(const void *const *inputs,
                                                                        void *const *outputs,
                                                                        void *workspace) {
  return {
      {
          {
              static_cast<const F *>(inputs[Input]),
              internal.input
          },
          {
              static_cast<const F *>(inputs[Offset]),
              internal.offset
          },
          {
              static_cast<const F *>(inputs[Mask]),
              internal.mask
          },
          {
              static_cast<const F *>(inputs[Weight]),
              internal.weight
          },
          {
              static_cast<const F *>(inputs[Bias]),
              internal.bias
          },
          {
              static_cast<F *>(workspace),
              internal.im2col_buffer
          },
      },

      {
          {
              static_cast<F *>(outputs[0]),
              internal.output
          },
      }
  };
}

int DCNLayerPlugin::enqueue(const PluginTensorDesc *inputDesc,
                            const PluginTensorDesc *outputDesc,
                            const void *const *inputs,
                            void *const *outputs,
                            void *workspace,
                            cudaStream_t stream) noexcept {

  DCNLayerExtra extra {cublas_context, 0};

  switch ((DataType) internal.data_type) {
    case DataType::kFLOAT: {
      auto [in, out] = makeView<float>(inputs, outputs, workspace);
      compute<float>(in, out, config, extra, stream);
      return 0;
    }

    case DataType::kHALF: {
      auto [in, out] = makeView<half>(inputs, outputs, workspace);
      compute<half>(in, out, config, extra, stream);
      return 0;
    }

    default:return 1;
  }
}

const PluginField DCNLayerPluginCreator::mPluginAttributes[]{
    {"stride", nullptr, PluginFieldType::kINT32, 2},
    {"padding", nullptr, PluginFieldType::kINT32, 2},
    {"dilation", nullptr, PluginFieldType::kINT32, 2},
    {"deformable_groups", nullptr, PluginFieldType::kINT32, 1},
    {"activation_type", nullptr, PluginFieldType::kINT32, 1},
    {"alpha", nullptr, PluginFieldType::kFLOAT32, 1},
    {"beta", nullptr, PluginFieldType::kFLOAT32, 1},
};

IPluginV2DynamicExt *DCNLayerPluginCreator::createPlugin(const char *name,
                                                         const PluginFieldCollection *fc) noexcept {
  if (fc->nbFields != mPFC.nbFields) {
    return nullptr;
  }

  DCNLayerConfig config{
      {1, 1},
      {1, 1},
      {1, 1},
      1,
      3,
      0.1,
      0
  };

  for (int32_t idx = 0; idx < fc->nbFields; ++idx) {
    const auto &field = fc->fields[idx];

    if (std::strcmp(field.name, "stride") == 0) {
      std::memcpy(&config.stride, field.data, sizeof(config.stride));
    }
    else if (std::strcmp(field.name, "padding") == 0) {
      std::memcpy(&config.padding, field.data, sizeof(config.padding));
    }
    else if (std::strcmp(field.name, "dilation") == 0) {
      std::memcpy(&config.dilation, field.data, sizeof(config.dilation));
    }
    else if (std::strcmp(field.name, "deformable_groups") == 0) {
      std::memcpy(&config.deformable_groups, field.data, sizeof(config.deformable_groups));
    }
    else if (std::strcmp(field.name, "activation_type") == 0) {
      std::memcpy(&config.activation_type, field.data, sizeof(config.activation_type));
    }
    else if (std::strcmp(field.name, "alpha") == 0) {
      std::memcpy(&config.alpha, field.data, sizeof(config.alpha));
    }
    else if (std::strcmp(field.name, "beta") == 0) {
      std::memcpy(&config.beta, field.data, sizeof(config.beta));
    }
  }

  auto p = new DCNLayerPlugin(&config);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

const PluginFieldCollection DCNLayerPluginCreator::mPFC{
    sizeof(DCNLayerPluginCreator::mPluginAttributes) / sizeof(PluginField),
    DCNLayerPluginCreator::mPluginAttributes,
};

const PluginFieldCollection *DCNLayerPluginCreator::getFieldNames() noexcept {
  return &nvinfer1::DCNLayerPluginCreator::mPFC;
}

IPluginV2DynamicExt *DCNLayerPluginCreator::deserializePlugin(const char *name,
                                                              const void *serialData,
                                                              size_t serialLength) noexcept {
  if (serialLength != sizeof(signature) + sizeof(DCNLayerConfig)) {
    return nullptr;
  }

  if (std::memcmp(&signature, serialData, sizeof(signature)) != 0) {
    return nullptr;
  }

  auto p = new DCNLayerPlugin((const char *) (serialData) + sizeof(signature), sizeof(DCNLayerConfig));
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

}
