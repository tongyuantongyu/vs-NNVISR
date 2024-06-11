//
// Created by TYTY on 2021-12-23 023.
//

#include <fstream>
#include <utility>

#include "cuda_runtime_api.h"

#include "inference.h"

//#include "debug/reveal.h"

#define CUDA_CHECK(status)                                                                                             \
  do {                                                                                                                 \
    auto ret = (status);                                                                                               \
    if (ret != 0) {                                                                                                    \
      std::stringstream s;                                                                                             \
      s << "Cuda failure at " __FILE__ ":" << __LINE__ << ": " << cudaGetErrorName(ret);                               \
      logger.log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, s.str().c_str());                                       \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (0)

#define COND_CHECK(cond, message)                                                                                      \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::stringstream s;                                                                                             \
      s << "Check failed " __FILE__ ":" << __LINE__ << ": " #cond ", " << message;                                     \
      logger.log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, s.str().c_str());                                       \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (0)

#define COND_CHECK_EMPTY(cond, message)                                                                                \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::stringstream s;                                                                                             \
      s << "Check failed " __FILE__ ":" << __LINE__ << ": " #cond ", " << message;                                     \
      logger.log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, s.str().c_str());                                       \
      return nullptr;                                                                                                  \
    }                                                                                                                  \
  } while (0)

static nvinfer1::ICudaEngine *loadModel(nvinfer1::IRuntime *runtime, nvinfer1::ILogger &logger,
                                        const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK_EMPTY(file.good(), "can't open engine file: " << path);

  file.seekg(0, std::ifstream::end);
  auto size = file.tellg();
  file.seekg(0, std::ifstream::beg);
  auto modelStream = std::make_unique<char[]>(size);
  COND_CHECK_EMPTY(modelStream, "Alloc " << size << " bytes failed.");
  file.read(modelStream.get(), size);
  file.close();

  auto engine = runtime->deserializeCudaEngine(modelStream.get(), size);
  COND_CHECK_EMPTY(runtime, "failed deserializing engine");

  return engine;
}

static std::string fe_engine_name(const InferenceConfig &config) {
  std::stringstream ss;
  ss << "fe_";
  ss << config.input_width << 'x' << config.input_height << '_' << config.scale_factor_w << "x" << config.scale_factor_h
     << "_b" << config.batch_extract << "_l" << config.extraction_layers;
  if (config.format == IOFormat::YUV420) {
    ss << "_yuv1-1";
  }
  if (config.use_fp16) {
    ss << "_fp16";
  }
  if (config.low_mem) {
    ss << "_lm";
  }
  ss << ".engine";
  return ss.str();
}

static std::string ff_engine_name(const InferenceConfig &config) {
  std::stringstream ss;
  ss << "ff_";
  ss << "n" << config.input_count;
  if (config.double_frame) {
    ss << "a";
  }
  if (config.extra_frame) {
    ss << "+";
  }
  ss << "_";
  ss << config.input_width << 'x' << config.input_height << '_' << config.scale_factor_w << "x" << config.scale_factor_h
     << "_b" << config.batch_fusion << "_l" << config.extraction_layers;
  if (config.format == IOFormat::YUV420) {
    ss << "_yuv1-1";
  }
  if (config.use_fp16) {
    ss << "_fp16";
  }
  if (config.low_mem) {
    ss << "_lm";
  }
  ss << ".engine";
  return ss.str();
}

InferenceContext::InferenceContext(InferenceConfig config, nvinfer1::ILogger &logger, std::filesystem::path path_prefix)
    : config(config), logger(logger),
      runtime(nvinfer1::createInferRuntime(logger)), path_prefix {std::move(path_prefix)}, engine {} {}

bool InferenceContext::has_file() {
  return exists(path_prefix / fe_engine_name(config)) && exists(path_prefix / ff_engine_name(config));
}

bool InferenceContext::load_engine() {
  engine.feature_extract = loadModel(runtime, logger, path_prefix / fe_engine_name(config));
  engine.feature_fusion = loadModel(runtime, logger, path_prefix / ff_engine_name(config));
  return good();
}

static constexpr void *ptr_add(void *b, size_t n) {
  return static_cast<uint8_t *>(b) + n;
}
static constexpr size_t alignment(size_t size, size_t alignment) {
  return (size + (alignment - 1)) & (~(alignment - 1));
}
static constexpr size_t ceil_half(size_t size) {
  return (size + 1) / 2;
}

InferenceSession::InferenceSession(InferenceContext &ctx)
    : ctx(ctx), context {ctx.engine.feature_extract->createExecutionContextWithoutDeviceMemory(),
                         ctx.engine.feature_fusion->createExecutionContextWithoutDeviceMemory()},
      last_batch {-1, -1}, last_offset_in {-1, -1}, last_offset_out {-1, -1}, good_ {false}, stream {nullptr},
      executionMemory {nullptr} {
  if (context.feature_extract == nullptr || context.feature_fusion == nullptr) {
    return;
  }

  auto &logger = ctx.logger;
  auto &config = ctx.config;

  CUDA_CHECK(cudaStreamCreate(&stream));

  const size_t eSize = config.use_fp16 ? 2 : 4;
  auto input_height = config.input_height;
  auto input_width = config.input_width;

  auto deviceMemory =
      std::max(ctx.engine.feature_extract->getDeviceMemorySize(), ctx.engine.feature_fusion->getDeviceMemorySize());
  size_t freeMemory {};
  cudaMemGetInfo(&freeMemory, nullptr);
  logger.log(freeMemory > deviceMemory ? nvinfer1::ILogger::Severity::kINFO : nvinfer1::ILogger::Severity::kWARNING,
             ("Device memory: " + std::to_string(freeMemory) + " bytes free, " + std::to_string(deviceMemory) +
              " bytes needed.")
                 .c_str());
  CUDA_CHECK(cudaMallocAsync(&executionMemory, deviceMemory, stream));
  context.feature_extract->setDeviceMemory(executionMemory);
  context.feature_fusion->setDeviceMemory(executionMemory);

  context.feature_extract->setOptimizationProfileAsync(0, stream);
  context.feature_fusion->setOptimizationProfileAsync(0, stream);

  features.resize(config.extraction_layers);

  size_t buffer_offset = 0;
  if (config.format == IOFormat::RGB) {
    cudaBuffers.resize(1 + config.extraction_layers + 1);

    shape_t<5> input_shape {config.batch_extract, 3, input_height, input_width, offset_t(eSize)};
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[0], input_shape.count(), stream));
    input = {reinterpret_cast<uint8_t *>(cudaBuffers[0]), input_shape};
    buffer_offset = 1;

    trace("alloc input RGB: " + describe(input));
  }
  else if (config.format == IOFormat::YUV420) {
    cudaBuffers.resize(2 + config.extraction_layers + 2);

    shape_t<5> input_shape {config.batch_extract, 1, input_height, input_width, offset_t(eSize)};
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[0], input_shape.count(), stream));
    input = {reinterpret_cast<uint8_t *>(cudaBuffers[0]), input_shape};
    shape_t<5> input_shape_uv {config.batch_extract, 2, offset_t(ceil_half(input_height)),
                               offset_t(ceil_half(input_width)), offset_t(eSize)};
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[1], input_shape_uv.count(), stream));
    input_uv = {reinterpret_cast<uint8_t *>(cudaBuffers[1]), input_shape_uv};
    buffer_offset = 2;

    trace("alloc input Y: " + describe(input) + ", UV: " + describe(input_uv));
  }

  auto layer_width = input_width;
  auto layer_height = input_height;
  for (int i = 0; i < config.extraction_layers; ++i) {
    int32_t feature_number = config.batch_extract + int(config.extra_frame);
    shape_t<5> feature_shape {feature_number, config.feature_count, layer_height, layer_width, offset_t(eSize)};
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[buffer_offset + i], feature_shape.count(), stream));
    features[i] = {reinterpret_cast<uint8_t *>(cudaBuffers[buffer_offset + i]), feature_shape};
    COND_CHECK(features[i].at(0).size() % 4 == 0, "bad feature size, try make it more aligned");
    layer_width = ceil_half(layer_width);
    layer_height = ceil_half(layer_height);

    trace("alloc feature " + std::to_string(i) + ": " + describe(features[i]));
  }

  buffer_offset += config.extraction_layers;

  if (config.format == IOFormat::RGB) {
    shape_t<7> output_shape {config.input_count,
                             config.double_frame ? 2 : 1,
                             config.batch_fusion,
                             3,
                             offset_t(double(input_height) * config.scale_factor_h),
                             offset_t(double(input_width) * config.scale_factor_w),
                             offset_t(eSize)};
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[buffer_offset], output_shape.count(), stream));
    output = {reinterpret_cast<uint8_t *>(cudaBuffers[buffer_offset]), output_shape};

    trace("alloc output RGB: " + describe(output));
  }
  else if (config.format == IOFormat::YUV420) {
    shape_t<7> output_shape {config.input_count,
                             config.double_frame ? 2 : 1,
                             config.batch_fusion,
                             1,
                             offset_t(double(input_height) * config.scale_factor_h),
                             offset_t(double(input_width) * config.scale_factor_w),
                             offset_t(eSize)};
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[buffer_offset], output_shape.count(), stream));
    output = {reinterpret_cast<uint8_t *>(cudaBuffers[buffer_offset]), output_shape};

    shape_t<7> output_shape_uv {config.input_count,
                                config.double_frame ? 2 : 1,
                                config.batch_fusion,
                                2,
                                offset_t(ceil_half(double(input_height) * config.scale_factor_h)),
                                offset_t(ceil_half(double(input_width) * config.scale_factor_w)),
                                offset_t(eSize)};
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[buffer_offset + 1], output_shape_uv.count(), stream));
    output_uv = {reinterpret_cast<uint8_t *>(cudaBuffers[buffer_offset + 1]), output_shape_uv};

    trace("alloc output Y: " + describe(output) + ", UV: " + describe(output_uv));
  }

  COND_CHECK(input.at(0).size() % 4 == 0, "bad input size, try make it more aligned");
  COND_CHECK(output.at(0, 0, 0).size() % 4 == 0, "bad input size, try make it more aligned");
  if (config.format == IOFormat::YUV420) {
    COND_CHECK(input_uv.at(0).size() % 4 == 0, "bad input size, try make it more aligned");
    COND_CHECK(output_uv.at(0, 0, 0).size() % 4 == 0, "bad input size, try make it more aligned");
  }

  good_ = true;
}

InferenceSession::~InferenceSession() {
  auto &logger = ctx.logger;
  if (stream == nullptr) {
    return;
  }

  if (executionMemory != nullptr) {
    CUDA_CHECK(cudaFreeAsync(executionMemory, stream));
  }

  for (auto *p: cudaBuffers) {
    if (p != nullptr) {
      CUDA_CHECK(cudaFreeAsync(p, stream));
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void InferenceSession::extractBatch(int32_t offset_in, int32_t offset_out, int32_t batch) {
  auto &logger = ctx.logger;
  auto &config = ctx.config;
  auto input_height = config.input_height;
  auto input_width = config.input_width;

  auto input_count = config.batch_extract + int(config.extra_frame);

  COND_CHECK(batch > 0, "invalid extract batch");
  COND_CHECK(offset_in + batch <= input_count, "invalid extract batch");
  COND_CHECK(offset_out + batch <= input_count + 1, "invalid extract batch");

  if (batch != last_batch.feature_extract) {
    trace("extract batch " + std::to_string(batch));
    if (config.format == IOFormat::RGB) {
      COND_CHECK(context.feature_extract->setInputShape("rgb", {4, {batch, 3, input_height, input_width}}),
                 "failed configure shape for input `rgb`.");
    }
    else if (config.format == IOFormat::YUV420) {
      COND_CHECK(context.feature_extract->setInputShape("y", {4, {batch, 1, input_height, input_width}}),
                 "failed configure shape for input `y`.");
      COND_CHECK(context.feature_extract->setInputShape(
                     "uv", {4, {batch, 2, int32_t(input_uv.shape[2]), int32_t(input_uv.shape[3])}}),
                 "failed configure shape for input `uv`.");
    }
    last_batch.feature_extract = batch;
  }

  if (offset_in != last_offset_in.feature_extract) {
    if (config.format == IOFormat::RGB) {
      COND_CHECK(context.feature_extract->setTensorAddress("rgb", input.at(offset_in).data),
                 "failed configure memory for input `rgb`.");
    }
    else if (config.format == IOFormat::YUV420) {
      trace("config extract input y: " + describe(input.at(offset_in)) + ", uv: " + describe(input_uv.at(offset_in)));
      COND_CHECK(context.feature_extract->setTensorAddress("y", input.at(offset_in).data),
                 "failed configure memory for input `y`.");
      COND_CHECK(context.feature_extract->setTensorAddress("uv", input_uv.at(offset_in).data),
                 "failed configure memory for input `uv`.");
    }

    last_offset_in.feature_extract = offset_in;
  }

  if (offset_out != last_offset_out.feature_extract) {
    for (int i = 0; i < config.extraction_layers; ++i) {
      auto name = "l" + std::to_string(i);
      trace("config extract output " + name + ": " + describe(features[i].at(offset_out)));
      COND_CHECK(context.feature_extract->setTensorAddress(name.c_str(), features[i].at(offset_out).data),
                 "failed configure memory for output `" << name << "`.");
    }

    last_offset_out.feature_extract = offset_out;
  }
}

void InferenceSession::fusionBatch(int32_t batch) {
  auto &logger = ctx.logger;
  auto &config = ctx.config;
  auto input_height = config.input_height;
  auto input_width = config.input_width;
  auto feature_count = config.feature_count;

  COND_CHECK(batch > 0, "invalid fusion batch");

  if (batch != last_batch.feature_fusion) {
    trace("fusion batch " + std::to_string(batch));
    auto layer_width = input_width;
    auto layer_height = input_height;
    for (int i = 0; i < config.extraction_layers; ++i) {
      for (int j = 0; j < config.input_count; ++j) {
        auto name = "f" + std::to_string((config.interpolation ? 2 : 1) * j) + "l" + std::to_string(i);
        COND_CHECK(
            context.feature_fusion->setInputShape(name.c_str(), {4, {batch, feature_count, layer_height, layer_width}}),
            "failed configure shape for input `" << name << "`.");
      }
      layer_width = ceil_half(layer_width);
      layer_height = ceil_half(layer_height);
    }
    last_batch.feature_fusion = batch;
    COND_CHECK(context.feature_fusion->inferShapes(0, nullptr) == 0, "model has extra unknown inputs");
  }

  if (last_offset_out.feature_fusion == -1) {
    int n_input = config.extra_frame ? (config.input_count - 1) : config.input_count;
    for (int j = 0; j < n_input; ++j) {
      for (int i = 0; i < 2; ++i) {
        auto name = "h" + std::to_string((config.interpolation ? 2 : 1) * j + (config.double_frame ? i : 1));

        if (config.format == IOFormat::RGB) {
          COND_CHECK(context.feature_fusion->setTensorAddress(name.c_str(), output.at(j, i, 0).data),
                     "failed configure memory for output `" << name << "`.");
        }
        else if (config.format == IOFormat::YUV420) {
          trace("config fusion output " + name + "_y: " + describe(output.at(j, i)) + ", " + name +
                "_uv: " + describe(output_uv.at(j, i)));
          COND_CHECK(context.feature_fusion->setTensorAddress((name + "_y").c_str(), output.at(j, i, 0).data),
                     "failed configure memory for output `" << name << "_y`.");
          COND_CHECK(context.feature_fusion->setTensorAddress((name + "_uv").c_str(), output_uv.at(j, i, 0).data),
                     "failed configure memory for output `" << name << "_uv`.");
        }

        if (!config.double_frame) {
          break;
        }
      }
    }

    last_offset_out.feature_fusion = 0;
  }
}

void InferenceSession::fusionGroupedOffset(int32_t group_idx) {
  auto &logger = ctx.logger;
  auto &config = ctx.config;

  auto group_frames = (config.input_count - int(config.extra_frame)) * config.batch_fusion;

  for (int i = 0; i < config.extraction_layers; ++i) {
    for (int j = 0; j < config.input_count; ++j) {
      int32_t offset;
      if (!config.extra_frame) {
        offset = config.batch_fusion * j;
      }
      else {
        // input_count = 4, batch_fusion = 2
        // v:I0
        // |v:I3
        // || v:I1
        // || | v:I2
        // 0361425
        if (j == 0) {
          offset = 0;
        }
        else if (j == config.input_count - 1) {
          offset = 1;
        }
        else {
          offset = config.batch_fusion * j + 1;
        }
      }

      auto name = "f" + std::to_string((config.interpolation ? 2 : 1) * j) + "l" + std::to_string(i);
      trace("config (grouped) fusion input " + name + ": " +
            describe(features[i].at(offset + group_idx * group_frames)) +
            "(" + std::to_string(j) + " mapped to " + std::to_string(offset + group_idx * group_frames) + ")");
      COND_CHECK(context.feature_fusion->setTensorAddress(name.c_str(),
                                                          features[i].at(offset + group_idx * group_frames).data),
                 "failed configure memory for input `" << name << "`.");
    }
  }
}

void InferenceSession::fusionCustomOffset(const std::vector<int32_t> &indexes) {
  auto &logger = ctx.logger;
  auto &config = ctx.config;

  assert(indexes.size() == config.input_count);

  for (int i = 0; i < config.extraction_layers; ++i) {
    for (int j = 0; j < config.input_count; ++j) {
      auto name = "f" + std::to_string((config.interpolation ? 2 : 1) * j) + "l" + std::to_string(i);
      trace("config (custom) fusion input " + name + ": " + describe(features[i].at(internalFeatureIndex(indexes[j]))) +
            "(" + std::to_string(indexes[j]) + " mapped to " + std::to_string(internalFeatureIndex(indexes[j])) + ")");
      COND_CHECK(
          context.feature_fusion->setTensorAddress(name.c_str(), features[i].at(internalFeatureIndex(indexes[j])).data),
          "failed configure memory for input `" << name << "`.");
    }
  }
}

int32_t InferenceSession::internalFeatureIndex(int32_t idx) {
  auto &config = ctx.config;

  if (config.extra_frame) {
    // input_count = 4, batch_fusion = 2
    // 0 3 6 9 C      0 1 2 7 8
    //   1 4 7 A  ->    3 4 9 A
    //   2 5 8 B        5 6 B C

    // 0361425 9C7A8B

    if (idx == 0) {
      return 0;
    }

    auto step = config.input_count - 1;
    auto batch = step * config.batch_fusion;
    auto row = idx % step;
    idx -= 1;
    auto page = idx / batch;
    auto column = (idx % batch) / step;
    return page * batch + row * config.batch_fusion + column + 1;
  }
  else {
    // input_count = 3, batch_fusion = 2
    // 0 3 6 9        0 1 6 7
    // 1 4 7 A    ->  2 3 8 9
    // 2 5 8 B        4 5 A B

    // 031425697A8B

    auto step = config.input_count;
    auto batch = step * config.batch_fusion;
    auto row = idx % step;
    auto page = idx / batch;
    auto column = (idx % batch) / step;
    return page * batch + column * config.batch_fusion + row;
  }
}

void InferenceSession::duplicateExtractOutput(int32_t from, int32_t to) {
  auto &logger = ctx.logger;
  auto &config = ctx.config;

  COND_CHECK(from <= config.batch_extract && to <= config.batch_extract, "invalid index");

  from = internalFeatureIndex(from);
  to = internalFeatureIndex(to);

  for (int i = 0; i < config.extraction_layers; ++i) {
    trace("copy feature " + std::to_string(i) + " from " + describe(features[i].at(from)) + " to " +
          describe(features[i].at(to)));
    CUDA_CHECK(cudaMemcpyAsync(features[i].at(to).data, features[i].at(from).data, features[i].at(from).size(),
                               cudaMemcpyDeviceToDevice, stream));
  }
}

shape_t<3> InferenceSession::outputIndex(offset_t idx) {
  auto &config = ctx.config;
  shape_t<3> result {};

  // if double_frame not set, only frames with odd index is available
  if (!config.double_frame && idx % 2 != 1) {
    return {-1, -1, -1};
  }

  result[1] = config.double_frame ? (idx % 2) : 0;
  idx /= 2;
  auto output_count = config.input_count - int(config.extra_frame);
  result[0] = idx % output_count;
  result[2] = idx / output_count;

  return result;
}

bool InferenceSession::extract() {
  trace("run extract");
  return context.feature_extract->enqueueV3(stream);
}

bool InferenceSession::fusion() {
  trace("run fusion");
  return context.feature_fusion->enqueueV3(stream);
}
