//
// Created by TYTY on 2021-12-23 023.
//

#include "optimize.h"

#include "NvOnnxParser.h"

#define COND_CHECK_EMPTY(cond, message)                                                                                \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::stringstream s;                                                                                             \
      s << "Check failed " __FILE__ ":" << __LINE__ << ": " #cond ", " << message;                                     \
      logger.log(nvinfer1::ILogger::Severity::kERROR, s.str().c_str());                                                \
      return -1;                                                                                                       \
    }                                                                                                                  \
  } while (0)

static void inspectNetwork(nvinfer1::INetworkDefinition *network) {
  auto count = network->getNbLayers();
  for (int i = 0; i < count; ++i) {
    auto layer = network->getLayer(i);
    auto i_count = layer->getNbInputs();
    std::cerr << "#" << i << ": " << layer->getName() << ", " << int32_t(layer->getType()) << ": ";

    std::cerr << "from {";
    for (int j = 0; j < i_count; ++j) {
      std::string name = layer->getInput(j)->getName();
      if (name.size() > 15) {
        name = std::to_string(atoi(name.c_str() + 15));
      }
      std::cerr << name;
      auto size = layer->getInput(j)->getDimensions();
      std::cerr << "(";
      for (int k = 0; k < size.nbDims; ++k) {
        std::cerr << size.d[k] << ",";
      }
      std::cerr << "\x08), ";
    }
    std::cerr << "\x08\x08}\n";
  }
}

// feature extract:
//   0   2   4   6
//         |
//   0   2   4   6 (3x)

// feature fusion
//   0   2   4 (3x)
//   2   4   6 (3x)
//       |
//   1   3   5

// main
//   0   2   4
//   1   3   5
//   2   4   6
//       |
// HR of input

static constexpr size_t ceil_half(size_t size) {
  return (size + 1) / 2;
}

static std::string model_name_suffix(const OptimizationConfig &config) {
  std::stringstream ss;
  ss << "_n" << config.input_count;
  ss << "_" << config.scale_factor_w << "x" << config.scale_factor_h << "_l" << config.extraction_layers;
  if (config.format == IOFormat::YUV420) {
    ss << "_yuv1-1";
  }
  ss << ".onnx";
  return ss.str();
}

static std::string fe_engine_name(const OptimizationConfig &config) {
  std::stringstream ss;
  ss << "fe_";
  ss << config.input_width.opt << 'x' << config.input_height.opt << '_' << config.scale_factor_w << "x"
     << config.scale_factor_h << "_b" << config.batch_extract.opt << "_l" << config.extraction_layers;
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

static std::string ff_engine_name(const OptimizationConfig &config) {
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
  ss << config.input_width.opt << 'x' << config.input_height.opt << '_' << config.scale_factor_w << "x"
     << config.scale_factor_h << "_b" << config.batch_fusion.opt << "_l" << config.extraction_layers;
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

// Eww...
template<typename T>
static std::unique_ptr<T> make_unique_from_raw(T *ptr) {
  return std::unique_ptr<T>(ptr);
}

nvinfer1::IBuilderConfig *OptimizationContext::prepareConfig() const {
  auto conf = builder->createBuilderConfig();
  if (config.use_fp16) {
    conf->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  conf->setFlag(nvinfer1::BuilderFlag::kTF32);
  conf->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
  conf->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
  // /usr/src/tensorrt/bin/trtexec --verbose --noDataTransfers --useCudaGraph --separateProfileRun --useSpinWait --nvtxMode=verbose --loadEngine=./mutual_cycle.engine --exportTimes=./mutual_cycle.timing.json --exportProfile=./mutual_cycle.profile.json --exportLayerInfo=./mutual_cycle.graph.json --timingCacheFile=./timing.cache --best --avgRuns=1000 "--shapes=lf0:1x64x180x270,lf1:1x64x180x270,lf2:1x64x180x270"
  conf->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
  conf->setTacticSources(conf->getTacticSources() &
                         ~nvinfer1::TacticSources(1u << int32_t(nvinfer1::TacticSource::kCUDNN)));
  conf->setTacticSources(conf->getTacticSources() |
                         nvinfer1::TacticSources(1u << int32_t(nvinfer1::TacticSource::kCUBLAS)));
  if (config.low_mem) {
    conf->setTacticSources(conf->getTacticSources() &
                           ~nvinfer1::TacticSources(1u << int32_t(nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS)));
  }

  if (cache != nullptr) {
    conf->setTimingCache(*cache, false);
  }

  return conf;
}

nvinfer1::INetworkDefinition *OptimizationContext::createNetwork() const {
  return builder->createNetworkV2(1u << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
}

OptimizationContext::OptimizationContext(OptimizationConfig config, nvinfer1::ILogger &logger,
                                         std::filesystem::path path_prefix_, std::filesystem::path path_engine_)
    : config(config), logger(logger), path_prefix(std::move(path_prefix_)), path_engine(std::move(path_engine_)),
      builder(nvinfer1::createInferBuilder(logger)), runtime(nvinfer1::createInferRuntime(logger)), cache(nullptr),
      prop {}, total_memory {} {
  if (!builder) {
    logger.log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, "Cannot create infer builder");
    return;
  }
  cudaMemGetInfo(nullptr, &total_memory);
  cudaGetDeviceProperties(&prop, 0);
  logger.log(nvinfer1::ILogger::Severity::kINFO,
             ("Device has " + std::to_string(total_memory) + " bytes memory.").c_str());

  if (builder->platformHasFastFp16() && !config.use_fp16) {
    // CUDA Architecture 6.1 (Pascal, GTX10xx series) does not have really useful FP16.
    if (prop.major != 6 || prop.minor != 1) {
      logger.log(nvinfer1::ILogger::Severity::kWARNING, "Fast FP16 is available but not enabled.");
    }
  }

  auto cache_file = path_engine / "timing.cache";
  std::ifstream input(cache_file, std::ios::binary | std::ios::in);
  auto conf = make_unique_from_raw(builder->createBuilderConfig());
  if (input.is_open()) {
    auto size = std::filesystem::file_size(cache_file);
    auto values = std::make_unique<char[]>(size);
    input.read(values.get(), size);
    cache.reset(conf->createTimingCache(values.get(), size));
    input.close();
  }
  if (cache == nullptr) {
    cache.reset(conf->createTimingCache(nullptr, 0));
  }
}

OptimizationContext::~OptimizationContext() {
  if (cache != nullptr) {
    std::ofstream output(path_engine / "timing.cache", std::ios::binary | std::ios::out);
    auto memory = cache->serialize();
    output.write(static_cast<char *>(memory->data()), memory->size());
    output.close();
  }
}

int OptimizationContext::optimize(const std::filesystem::path &folder) {
  auto fe_target = path_engine / folder / fe_engine_name(config);
  if (!exists(fe_target)) {
    auto fe_source_file = path_prefix / "models" / folder / ("fe" + model_name_suffix(config));
    std::ifstream input_fe(fe_source_file, std::ios::binary | std::ios::in);
    COND_CHECK_EMPTY(input_fe.is_open(), "Source model file not exist:" << fe_source_file);
    std::vector<uint8_t> fe_source(std::filesystem::file_size(fe_source_file));
    input_fe.read((char *) (fe_source.data()), fe_source.size());
    auto ret = buildFeatureExtract(std::move(fe_source), fe_target);
    if (ret != 0) {
      return ret;
    }
  }

  auto ff_target = path_engine / folder / ff_engine_name(config);
  if (!exists(ff_target)) {
    auto ff_source_file = path_prefix / "models" / folder / ("ff" + model_name_suffix(config));
    std::ifstream input_ff(ff_source_file, std::ios::binary | std::ios::in);
    COND_CHECK_EMPTY(input_ff.is_open(), "Source model file not exist:" << ff_source_file);
    std::vector<uint8_t> ff_source(std::filesystem::file_size(ff_source_file));
    input_ff.read((char *) (ff_source.data()), ff_source.size());
    auto ret = buildFeatureFusion(std::move(ff_source), ff_target);
    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

int OptimizationContext::buildFeatureExtract(std::vector<uint8_t> input, const std::filesystem::path &output) {
  auto network = make_unique_from_raw(createNetwork());
  auto profile = builder->createOptimizationProfile();
  auto parser = make_unique_from_raw(nvonnxparser::createParser(*network, logger));
  COND_CHECK_EMPTY(parser->parse(input.data(), input.size()), "Failed parse source model.");
  input.clear();

  auto ioDataType = config.use_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;

  if (config.format == IOFormat::RGB) {
    profile->setDimensions(
        "rgb", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4 {config.batch_extract.min, 3, config.input_height.min, config.input_width.min});
    profile->setDimensions(
        "rgb", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4 {config.batch_extract.opt, 3, config.input_height.opt, config.input_width.opt});
    profile->setDimensions(
        "rgb", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4 {config.batch_extract.max, 3, config.input_height.max, config.input_width.max});
  }
  else {
    profile->setDimensions(
        "y", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4 {config.batch_extract.min, 1, config.input_height.min, config.input_width.min});
    profile->setDimensions(
        "y", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4 {config.batch_extract.opt, 1, config.input_height.opt, config.input_width.opt});
    profile->setDimensions(
        "y", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4 {config.batch_extract.max, 1, config.input_height.max, config.input_width.max});

    profile->setDimensions(
        "uv", nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4 {config.batch_extract.min, 2, config.input_height.min / 2, config.input_width.min / 2});
    profile->setDimensions(
        "uv", nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4 {config.batch_extract.opt, 2, config.input_height.opt / 2, config.input_width.opt / 2});
    profile->setDimensions(
        "uv", nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4 {config.batch_extract.max, 2, config.input_height.max / 2, config.input_width.max / 2});
  }

  for (int i = 0; i < network->getNbInputs(); ++i) {
    network->getInput(i)->setType(ioDataType);
  }

  for (int i = 0; i < network->getNbOutputs(); ++i) {
    network->getOutput(i)->setType(ioDataType);
  }
  logger.log(nvinfer1::ILogger::Severity::kINFO, "Done define feature extract net.");

  auto optimize_config = make_unique_from_raw(prepareConfig());
  // value from experience
  //  optimize_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, total_memory / 24);
  optimize_config->addOptimizationProfile(profile);
  auto modelStream = make_unique_from_raw(builder->buildSerializedNetwork(*network, *optimize_config));
  COND_CHECK_EMPTY(modelStream != nullptr, "Failed build feature extract net.");
  logger.log(nvinfer1::ILogger::Severity::kINFO, "Done build feature extract net.");

  auto parent = output;
  std::filesystem::create_directories(parent.remove_filename());
  std::ofstream p(output, std::ios::binary);
  COND_CHECK_EMPTY(p.is_open(), "Unable to open engine file for output.");
  p.write(static_cast<const char *>(modelStream->data()), modelStream->size());
  p.close();

  auto engine = make_unique_from_raw(runtime->deserializeCudaEngine(modelStream->data(), modelStream->size()));
  auto inspector = make_unique_from_raw(engine->createEngineInspector());
  auto context = make_unique_from_raw(engine->createExecutionContextWithoutDeviceMemory());
  context->setOptimizationProfileAsync(0, nullptr);
  cudaStreamSynchronize(nullptr);

  if (config.format == IOFormat::RGB) {
    context->setInputShape(
        "rgb", nvinfer1::Dims4 {config.batch_extract.opt, 3, config.input_height.opt, config.input_width.opt});
  }
  else {
    context->setInputShape(
        "y", nvinfer1::Dims4 {config.batch_extract.opt, 1, config.input_height.opt, config.input_width.opt});
    context->setInputShape(
        "uv", nvinfer1::Dims4 {config.batch_extract.opt, 2, config.input_height.opt / 2, config.input_width.opt / 2});
  }
  COND_CHECK_EMPTY(context->inferShapes(0, nullptr) == 0, "Unknown extra input found");
  inspector->setExecutionContext(context.get());
  auto path_layers = output;
  path_layers.replace_extension(".layers.json");
  std::ofstream info(path_layers, std::ios::binary);
  info << inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
  info.close();

  logger.log(nvinfer1::ILogger::Severity::kINFO, "Done save feature extract net.");

  return 0;
}

int OptimizationContext::buildFeatureFusion(std::vector<uint8_t> input, const std::filesystem::path &output) {
  auto network = make_unique_from_raw(createNetwork());
  auto profile = builder->createOptimizationProfile();
  auto parser = make_unique_from_raw(nvonnxparser::createParser(*network, logger));
  COND_CHECK_EMPTY(parser->parse(input.data(), input.size()), "Failed parse source model.");
  input.clear();

  auto ioDataType = config.use_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
  auto layer_height = config.input_height;
  auto layer_width = config.input_width;

  for (int i = 0; i < config.extraction_layers; ++i) {
    for (int j = 0; j < config.input_count; ++j) {
      auto name = "f" + std::to_string((config.interpolation ? 2 : 1) * j) + "l" + std::to_string(i);
      profile->setDimensions(
          name.c_str(), nvinfer1::OptProfileSelector::kMIN,
          nvinfer1::Dims4 {config.batch_fusion.min, config.feature_count, layer_height.min, layer_width.min});
      profile->setDimensions(
          name.c_str(), nvinfer1::OptProfileSelector::kOPT,
          nvinfer1::Dims4 {config.batch_fusion.opt, config.feature_count, layer_height.opt, layer_width.opt});
      profile->setDimensions(
          name.c_str(), nvinfer1::OptProfileSelector::kMAX,
          nvinfer1::Dims4 {config.batch_fusion.max, config.feature_count, layer_height.max, layer_width.max});
    }
    layer_height.min = ceil_half(layer_height.min);
    layer_height.opt = ceil_half(layer_height.opt);
    layer_height.max = ceil_half(layer_height.max);
    layer_width.min = ceil_half(layer_width.min);
    layer_width.opt = ceil_half(layer_width.opt);
    layer_width.max = ceil_half(layer_width.max);
  }

  for (int i = 0; i < network->getNbInputs(); ++i) {
    network->getInput(i)->setType(ioDataType);
  }

  for (int i = 0; i < network->getNbOutputs(); ++i) {
    network->getOutput(i)->setType(ioDataType);
  }
  logger.log(nvinfer1::ILogger::Severity::kINFO, "Done define feature fusion net.");

  auto optimize_config = make_unique_from_raw(prepareConfig());
  // value from experience
  //  optimize_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, total_memory / 2);
  optimize_config->addOptimizationProfile(profile);
  auto modelStream = make_unique_from_raw(builder->buildSerializedNetwork(*network, *optimize_config));
  COND_CHECK_EMPTY(modelStream != nullptr, "Failed build feature fusion net.");
  logger.log(nvinfer1::ILogger::Severity::kINFO, "Done build feature fusion net.");

  auto parent = output;
  std::filesystem::create_directories(parent.remove_filename());
  std::ofstream p(output, std::ios::binary);
  COND_CHECK_EMPTY(p.is_open(), "Unable to open engine file for output.");
  p.write(static_cast<const char *>(modelStream->data()), modelStream->size());
  p.close();

  auto engine = make_unique_from_raw(runtime->deserializeCudaEngine(modelStream->data(), modelStream->size()));
  auto inspector = make_unique_from_raw(engine->createEngineInspector());
  auto context = make_unique_from_raw(engine->createExecutionContextWithoutDeviceMemory());
  context->setOptimizationProfileAsync(0, nullptr);
  cudaStreamSynchronize(nullptr);

  auto opt_height = config.input_height.opt;
  auto opt_width = config.input_width.opt;
  for (int i = 0; i < config.extraction_layers; ++i) {
    for (int j = 0; j < config.input_count; ++j) {
      auto name = "f" + std::to_string((config.interpolation ? 2 : 1) * j) + "l" + std::to_string(i);
      context->setInputShape(
          name.c_str(), nvinfer1::Dims4 {config.batch_fusion.opt, config.feature_count, opt_height, opt_width});
    }
    opt_height = ceil_half(opt_height);
    opt_width = ceil_half(opt_width);
  }
  COND_CHECK_EMPTY(context->inferShapes(0, nullptr) == 0, "Unknown extra input found");
  inspector->setExecutionContext(context.get());
  auto path_layers = output;
  path_layers.replace_extension(".layers.json");
  std::ofstream info(path_layers, std::ios::binary);
  info << inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
  info.close();

  logger.log(nvinfer1::ILogger::Severity::kINFO, "Done save feature fusion net.");

  return 0;
}
