#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "NvInferRuntime.h"

#include "config.h"
#include "md_view.h"
#include "utils.h"

template<class T>
struct ModelStuff {
  T feature_extract;
  T feature_fusion;
};

class InferenceSession;

class InferenceContext {
  nvinfer1::ILogger &logger;
  nvinfer1::IRuntime *runtime;
  std::filesystem::path path_prefix;
  ModelStuff<nvinfer1::ICudaEngine *> engine;

  friend class InferenceSession;

 public:
  InferenceConfig config;
  InferenceContext(InferenceConfig config, nvinfer1::ILogger &logger, std::filesystem::path path_prefix);
  bool has_file();
  bool load_engine();

  bool good() { return runtime != nullptr && engine.feature_extract != nullptr && engine.feature_fusion != nullptr; }
};

class InferenceSession {
  InferenceContext ctx;

  ModelStuff<nvinfer1::IExecutionContext *> context;
  std::vector<void *> cudaBuffers;
  void *executionMemory;
  ModelStuff<int32_t> last_batch, last_offset_in, last_offset_out;
  bool good_;

  void trace(const std::string &info) {
    // no fold
//     ctx.logger.log(nvinfer1::ILogger::Severity::kINFO, ("Infer Trace: " + info).c_str());
  }

 public:
  cudaStream_t stream;

  md_view<uint8_t, 5> input, input_uv;
  md_view<uint8_t, 7> output, output_uv;
  std::vector<md_view<uint8_t, 5>> features;

  explicit InferenceSession(InferenceContext &ctx);
  ~InferenceSession();

  bool good() const { return good_; }

  void extractBatch(int32_t offset_in, int32_t offset_out, int32_t batch);
  void fusionBatch(int32_t batch);
  void fusionGroupedOffset(int32_t group_idx);
  void fusionCustomOffset(const std::vector<int32_t> &indexes);

  void duplicateExtractOutput(int32_t from, int32_t to);

  int32_t internalFeatureIndex(int32_t idx);
  shape_t<3> outputIndex(offset_t idx);

  bool extract();
  bool fusion();
};
