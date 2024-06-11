#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>

#include <unordered_map>

#if defined(_MSC_VER) && !defined(IS_CONDA_BUILD)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <libloaderapi.h>
#endif

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <cuda_fp16.h>

#include "inference.h"
#include "layers.h"
#include "optimize.h"
#include "reformat.h"

#include "md_view.h"

#include "VSConstants4.h"
#include "VSHelper4.h"
#include "VapourSynth4.h"

// ---------------------------------------------------------------------------------------------------------------------
// Utils

class Logger : public nvinfer1::ILogger {
  typedef void(VS_CC *logMessage_t)(int msgType, const char *msg, VSCore *core) VS_NOEXCEPT;

 public:
  Logger(VSCore *core, logMessage_t logMessage) : core(core), logMessage(logMessage) {}
  void log(Severity severity, const char *message) noexcept override {
    auto severity_int = int32_t(severity);
    if (severity_int < 0 || severity_int > 4) {
      severity_int = 0;
    }
    if (severity == nvinfer1::ILogger::Severity::kINFO) {
      const std::string_view message_view = message;
      for (const auto &blocked: blockedMessages) {
        if (message_view.starts_with(blocked)) {
          severity_int = int32_t(nvinfer1::ILogger::Severity::kVERBOSE);
          break;
        }
      }
    }
    logMessage(typeMap[severity_int], message, core);
  }

  void log(Severity severity, const std::string_view &message) noexcept {
    auto severity_int = int32_t(severity);
    if (severity_int < 0 || severity_int > 4) {
      severity_int = 0;
    }
    logMessage(typeMap[severity_int], message.data(), core);
  }

 private:
  VSCore *core;
  logMessage_t logMessage;

#if defined(NDEBUG) || defined(_NDEBUG)
  constexpr static VSMessageType trtInfoLevel = VSMessageType::mtDebug;
#else
  constexpr static VSMessageType trtInfoLevel = VSMessageType::mtInformation;
#endif

  constexpr static VSMessageType typeMap[] = {VSMessageType::mtFatal, VSMessageType::mtWarning,
                                              VSMessageType::mtWarning, trtInfoLevel, VSMessageType::mtDebug};

  constexpr static std::string_view blockedMessages[] = {
      "No importer registered for op: ",
      "Searching for plugin: ",
      "Successfully created plugin: ",
  };
};

struct scale_ratios_t {
  struct fma {
    float a;
    float b;
  };

  union {
    fma z[3];
    struct {
      fma y, u, v;
    };
    struct {
      fma r, g, b;
    };
  };
};

//const std::array<float, 3> default_norm_mean = {0.485, 0.456, 0.406};
//const std::array<float, 3> default_norm_std = {0.229, 0.224, 0.225};
const std::array<float, 3> default_norm_mean = {0, 0, 0};
const std::array<float, 3> default_norm_std = {1, 1, 1};

struct color_space_t {
  VSColorPrimaries cp;
  VSTransferCharacteristics tc;
  VSMatrixCoefficients mc;
  VSColorRange r;
};

template<size_t N>
static int getFloatArray(const char *name, std::array<float, N> &data, const VSMap *in, const VSAPI *vsapi,
                         const std::array<float, N> &def) {
  int err;

  vsapi->mapGetFloatArray(in, name, &err);
  if (err) {
    data = def;
    return 0;
  }

  for (int i = 0; i < N; ++i) {
    data[i] = vsapi->mapGetFloatSaturated(in, name, i, &err);
    if (err) {
      return -1;
    }
  }

  return 0;
}

static size_t alignment(size_t size, size_t alignment) {
  return (size + (alignment - 1)) & (~(alignment - 1));
}

static const char *safe_cstr(const char *str) {
  return str ? str : "";
}

struct colorPrimariesEntry {
  VSColorPrimaries colorPrimariesEnum;
  float primaries[8];// rX, rY, gX, gY, bX, bY, wX, wY
};

const std::array<colorPrimariesEntry, 11> colorPrimariesTable {
    {{VSC_PRIMARIES_BT709, {0.64f, 0.33f, 0.3f, 0.6f, 0.15f, 0.06f, 0.3127f, 0.329f}},
     {VSC_PRIMARIES_BT470_M, {0.67f, 0.33f, 0.21f, 0.71f, 0.14f, 0.08f, 0.310f, 0.316f}},
     {VSC_PRIMARIES_BT470_BG, {0.64f, 0.33f, 0.29f, 0.60f, 0.15f, 0.06f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_ST170_M, {0.630f, 0.340f, 0.310f, 0.595f, 0.155f, 0.070f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_ST240_M, {0.630f, 0.340f, 0.310f, 0.595f, 0.155f, 0.070f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_FILM, {0.681f, 0.319f, 0.243f, 0.692f, 0.145f, 0.049f, 0.310f, 0.316f}},
     {VSC_PRIMARIES_BT2020, {0.708f, 0.292f, 0.170f, 0.797f, 0.131f, 0.046f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_ST428, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.3333f, 0.3333f}},
     {VSC_PRIMARIES_ST431_2, {0.680f, 0.320f, 0.265f, 0.690f, 0.150f, 0.060f, 0.314f, 0.351f}},
     {VSC_PRIMARIES_ST432_1, {0.680f, 0.320f, 0.265f, 0.690f, 0.150f, 0.060f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_EBU3213_E, {0.630f, 0.340f, 0.295f, 0.605f, 0.155f, 0.077f, 0.3127f, 0.3290f}}}};

struct matrixCoefficientsEntry {
  VSMatrixCoefficients matrixCoefficientsEnum;
  const float kr;
  const float kb;
};

const std::array<matrixCoefficientsEntry, 6> matrixCoefficientsTable {{{VSC_MATRIX_BT709, 0.2126f, 0.0722f},
                                                                       {VSC_MATRIX_FCC, 0.30f, 0.11f},
                                                                       {VSC_MATRIX_BT470_BG, 0.299f, 0.114f},
                                                                       {VSC_MATRIX_ST170_M, 0.299f, 0.114f},
                                                                       {VSC_MATRIX_ST240_M, 0.212f, 0.087f},
                                                                       {VSC_MATRIX_BT2020_NCL, 0.2627f, 0.0593f}}};

// ---------------------------------------------------------------------------------------------------------------------
// Filter

class NNVISRFilter {
  int num_frames;
  int extract_begin;
  int extract_end;
  int fusion_in_begin;
  int fusion_out_begin;
  bool end_of_scene;
  bool start_of_batch, start_of_scene;

  int last_requested_frame;
  int last_output_frame;

  VSColorFamily color_family;
  std::filesystem::path model_path;
  std::filesystem::path engine_path;
  std::filesystem::path model;
  InferenceConfig config;
  InferenceContext *ctx;
  InferenceSession *session;
  const VSFrame *first_frame;
  bool raw_norm;
  scale_ratios_t norm, denorm;
  Logger *logger;
  float y_min, y_max, uv_min, uv_max;
  uint8_t *ioBuffer[2] {};
  uint8_t *ioPointer[6];
  shape_t<2> input_shape_y, input_shape_uv, output_shape_y, output_shape_uv;
  shape_t<2> input_tensor_y, input_tensor_uv, output_tensor_y, output_tensor_uv;

  //  std::unordered_map<const VSFrame *, int> frame_idx;

  template<class F, class U>
  std::string readPlane(md_view<F, 2> dst, md_uview<const U, 2> src, md_view<U, 2> cuda_tmp, float a, float b);
  template<class F, class U>
  std::string writePlane(md_uview<U, 2> dst, md_view<const F, 2> src, md_view<U, 2> cuda_tmp, float a, float b,
                         float min, float max);
  template<class U>
  std::string uploadYUV(offset_t position, const VSFrame *frame, const VSAPI *vsapi);
  template<class U>
  std::string downloadYUV(offset_t position, VSFrame *frame, const VSAPI *vsapi);
  template<class U>
  std::string uploadRGB(offset_t position, const VSFrame *frame, const VSAPI *vsapi);
  template<class U>
  std::string downloadRGB(offset_t position, VSFrame *frame, const VSAPI *vsapi);

  void trace(const std::string &info) {
    // no fold
    //         logger->log(Logger::Severity::kWARNING, "NNVISR Trace: " + info);
  }

 public:
  VSNode *node;
  VSVideoInfo vi, vo;
  std::vector<const VSFrame *> requested_frames;
  std::string init1(const VSMap *in, VSCore *core, const VSAPI *vsapi);
  std::string init2(const VSFrame *frame, VSCore *core, const VSAPI *vsapi);
  std::string requestFrames(int n, VSFrameContext *frameCtx, const VSAPI *vsapi);
  std::string prepareFrame(int n, VSFrameContext *frameCtx, const VSAPI *vsapi);
  std::string extractFrame(int n, const VSFrame *&frame, VSCore *core, const VSAPI *vsapi);
  ~NNVISRFilter();

  std::string synchronize() {
    auto err = cudaStreamSynchronize(session->stream);
    if (err != cudaSuccess) {
      return std::string("NNVISR: failed synchronize CUDA stream: ") + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    return "";
  }
};

std::string NNVISRFilter::init1(const VSMap *in, VSCore *core, const VSAPI *vsapi) {
  int err;

  logger = new Logger {core, vsapi->logMessage};

  // Get a clip reference from the input arguments. This must be freed later.
  node = vsapi->mapGetNode(in, "clip", 0, nullptr);
  vi = *vsapi->getVideoInfo(node);
  num_frames = vi.numFrames;
  extract_begin = 0;
  extract_end = 0;
  //  trace("extract_end at " + std::to_string(extract_end));
  end_of_scene = true;
  last_output_frame = -1;

  if (!vsh::isConstantVideoFormat(&vi)) {
    return "NNVISR: only constant format input supported";
  }

  IOFormat format;
  if (vi.format.colorFamily == VSColorFamily::cfRGB) {
    format = IOFormat::RGB;
  }
  else if (vi.format.colorFamily == VSColorFamily::cfYUV) {
    if (vi.format.subSamplingH == 1 && vi.format.subSamplingW == 1) {
      format = IOFormat::YUV420;
    }
    else {
      return "NNVISR: only support 4:2:0 for YUV IO";
    }
  }
  else {
    return "NNVISR: only support RGB or YUV format";
  }
  color_family = VSColorFamily(vi.format.colorFamily);

  std::array<float, 3> tmp {};
  err = getFloatArray("norm_mean", tmp, in, vsapi, default_norm_mean);
  if (err) {
    return "NNVISR: norm_mean should have 3 values";
  }
  denorm.r.b = tmp[0];
  denorm.g.b = tmp[1];
  denorm.b.b = tmp[2];

  err = getFloatArray("norm_std", tmp, in, vsapi, default_norm_std);
  if (err) {
    return "NNVISR: norm_std should have 3 values";
  }
  denorm.r.a = tmp[0];
  denorm.g.a = tmp[1];
  denorm.b.a = tmp[2];

  auto input_count = int32_t(vsapi->mapGetInt(in, "input_count", 0, &err));
  if (err) {
    input_count = 1;
  }
  else if (input_count < 1) {
    return "NNVISR: input_count should >= 1";
  }

  auto feature_count = int32_t(vsapi->mapGetInt(in, "feature_count", 0, &err));
  if (err) {
    feature_count = 64;
  }
  else if (feature_count < 1) {
    return "NNVISR: feature_count should >= 1";
  }
  else if (feature_count % 8 != 0) {
    vsapi->logMessage(mtWarning, "NNVISR: feature_count not multiple of 8, this model can be inefficient.", core);
  }

  auto extraction_layers = int32_t(vsapi->mapGetInt(in, "extraction_layers", 0, &err));
  if (err) {
    extraction_layers = 1;
  }
  else if (extraction_layers < 1) {
    return "NNVISR: extraction_layers should >= 1";
  }

  auto extra_frame = bool(vsapi->mapGetInt(in, "extra_frame", 0, &err));
  if (err) {
    extra_frame = false;
  }

  auto double_frame = bool(vsapi->mapGetInt(in, "double_frame", 0, &err));
  if (err) {
    double_frame = false;
  }

  bool interpolation = bool(vsapi->mapGetInt(in, "interpolation", 0, &err));
  if (err) {
    interpolation = double_frame || extra_frame;
  }
  if (double_frame && !interpolation) {
    return "NNVISR: interpolation must be True if double_frame is True";
  }

  auto batch_size_fusion = int32_t(vsapi->mapGetInt(in, "batch_size_fusion", 0, &err));
  if (err) {
    batch_size_fusion = 1;
  }
  else if (batch_size_fusion < 1) {
    return "NNVISR: batch_size_fusion should >= 1";
  }

  auto batch_size = int32_t(vsapi->mapGetInt(in, "batch_size_extract", 0, &err));
  if (err) {
    batch_size = batch_size_fusion * (extra_frame ? (input_count - 1) : input_count);
  }
  else if (batch_size < 1) {
    return "NNVISR: batch_size should >= 1";
  }

  if (extra_frame) {
    if (batch_size % (batch_size_fusion * (input_count - 1)) != 0) {
      return "NNVISR: batch_size should be a multiple of batch_size_fusion * (input_count - 1)";
    }
  }
  else {
    if (batch_size % (batch_size_fusion * input_count) != 0) {
      return "NNVISR: batch_size should be a multiple of batch_size_fusion * input_count";
    }
  }

  auto scale_factor = float(vsapi->mapGetFloat(in, "scale_factor", 0, nullptr));
  if (scale_factor <= 0) {
    return "NNVISR: scale_factor should > 0";
  }

  auto scale_factor_h = float(vsapi->mapGetFloat(in, "scale_factor_h", 0, &err));
  if (err) {
    scale_factor_h = scale_factor;
  }
  else if (scale_factor_h <= 0) {
    return "NNVISR: scale_factor_h should > 0";
  }
  if (interpolation && !double_frame && (scale_factor != 1 || scale_factor_h != 1)) {
    return "NNVISR: if interpolation not producing double_frame, scale_factor must be 1";
  }

  auto use_fp16 = bool(vsapi->mapGetInt(in, "use_fp16", 0, &err));
  if (err) {
    use_fp16 = false;
  }

  raw_norm = bool(vsapi->mapGetInt(in, "raw_norm", 0, &err));
  if (err) {
    raw_norm = false;
  }

  model_path = safe_cstr(vsapi->mapGetData(in, "model_path", 0, &err));
  if (err) {
    model_path = vsapi->getPluginPath(vsapi->getPluginByID("dev.tyty.aim.nnvisr", core));
    model_path = model_path.remove_filename() / "dev.tyty.aim.nnvisr";
  }

  engine_path = safe_cstr(vsapi->mapGetData(in, "engine_path", 0, &err));
  if (err) {
    cudaDeviceProp prop {};
    cudaGetDeviceProperties(&prop, 0);
    engine_path = model_path / "engines" / std::to_string(getInferLibVersion()) / prop.name;
  }

  std::string model_name = safe_cstr(vsapi->mapGetData(in, "model", 0, &err));
  if (err) {
    model = ".";
  }
  else {
    model = model_name;
  }

  auto low_mem = bool(vsapi->mapGetInt(in, "low_mem", 0, &err));
  if (err) {
    low_mem = false;
  }

  config = {int32_t(vi.width), int32_t(vi.height), batch_size,  batch_size_fusion, input_count,
            feature_count,     extraction_layers,  extra_frame, double_frame,      interpolation,
            scale_factor,      scale_factor_h,     format,      use_fp16,          low_mem};

  vo = vi;
  vo.width = int(double(scale_factor) * vo.width);
  vo.height = int(double(scale_factor_h) * vo.height);
  if (interpolation) {
    vo.numFrames *= 2;
    vo.fpsNum *= 2;
    vsh::reduceRational(&vo.fpsNum, &vo.fpsDen);
  }

  auto out_format = vsapi->mapGetInt(in, "out_format", 0, &err);
  if (!err) {
    if (!vsapi->getVideoFormatByID(&vo.format, out_format, core)) {
      return "NNVISR: invalid out_format";
    }

    if (vo.format.colorFamily != vi.format.colorFamily || vo.format.subSamplingW != vi.format.subSamplingW ||
        vo.format.subSamplingH != vi.format.subSamplingH) {
      return "NNVISR: incompatible out_format: Mistach color family or chroma subsampling";
    }
  }

  if (color_family == VSColorFamily::cfRGB) {
    size_t input_y_size = alignment((size_t) vi.width * vi.height * vi.format.bytesPerSample, 4096);
    input_shape_y = {vi.height, vi.width};
    input_tensor_y = {config.input_height, config.input_width};

    size_t output_y_size = alignment((size_t) vo.width * vo.height * vo.format.bytesPerSample, 4096);
    output_shape_y = {vo.height, vo.width};
    output_tensor_y = {offset_t(double(config.input_height) * scale_factor),
                       offset_t(double(config.input_width) * scale_factor)};

    err = cudaMalloc((void **) &ioBuffer[0], 3 * input_y_size);
    if (err != cudaSuccess) {
      return "NNVISR: failed alloc " + std::to_string(3 * input_y_size) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    err = cudaMalloc((void **) &ioBuffer[1], 3 * output_y_size);
    if (err != cudaSuccess) {
      return "NNVISR: failed alloc " + std::to_string(3 * output_y_size) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    ioPointer[0] = ioBuffer[0];
    ioPointer[1] = ioBuffer[0] + input_y_size;
    ioPointer[2] = ioBuffer[0] + 2 * input_y_size;
    ioPointer[3] = ioBuffer[1];
    ioPointer[4] = ioBuffer[1] + output_y_size;
    ioPointer[5] = ioBuffer[1] + 2 * output_y_size;
  }
  else {
    size_t input_y_size = alignment((size_t) vi.width * vi.height * vi.format.bytesPerSample, 4096);
    int32_t input_uv_width = (vi.width + (1 << vi.format.subSamplingW) - 1) >> vi.format.subSamplingW;
    int32_t input_uv_height = (vi.height + (1 << vi.format.subSamplingH) - 1) >> vi.format.subSamplingH;
    size_t input_uv_size = alignment((size_t) input_uv_width * input_uv_height * vi.format.bytesPerSample, 4096);
    input_shape_y = {vi.height, vi.width};
    input_shape_uv = {input_uv_height, input_uv_width};
    input_tensor_y = {config.input_height, config.input_width};
    input_tensor_uv = {(input_tensor_y[0] + (1 << vo.format.subSamplingH) - 1) >> vo.format.subSamplingH,
                       (input_tensor_y[1] + (1 << vo.format.subSamplingW) - 1) >> vo.format.subSamplingW};

    size_t output_y_size = alignment((size_t) vo.width * vo.height * vo.format.bytesPerSample, 4096);
    int32_t output_uv_width = (vo.width + (1 << vo.format.subSamplingW) - 1) >> vo.format.subSamplingW;
    int32_t output_uv_height = (vo.height + (1 << vo.format.subSamplingH) - 1) >> vo.format.subSamplingH;
    size_t output_uv_size = alignment((size_t) output_uv_width * output_uv_height * vo.format.bytesPerSample, 4096);
    output_shape_y = {vo.height, vo.width};
    output_shape_uv = {output_uv_height, output_uv_width};
    output_tensor_y = {offset_t(double(config.input_height) * scale_factor),
                       offset_t(double(config.input_width) * scale_factor)};
    output_tensor_uv = {(output_tensor_y[0] + (1 << vo.format.subSamplingH) - 1) >> vo.format.subSamplingH,
                        (output_tensor_y[1] + (1 << vo.format.subSamplingW) - 1) >> vo.format.subSamplingW};

    err = cudaMalloc((void **) &ioBuffer[0], input_y_size + 2 * input_uv_size);
    if (err != cudaSuccess) {
      return "NNVISR: failed alloc " + std::to_string(input_y_size + 2 * input_uv_size) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    err = cudaMalloc((void **) &ioBuffer[1], output_y_size + 2 * output_uv_size);
    if (err != cudaSuccess) {
      return "NNVISR: failed alloc " + std::to_string(output_y_size + 2 * output_uv_size) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    ioPointer[0] = ioBuffer[0];
    ioPointer[1] = ioBuffer[0] + input_y_size;
    ioPointer[2] = ioBuffer[0] + input_y_size + input_uv_size;
    ioPointer[3] = ioBuffer[1];
    ioPointer[4] = ioBuffer[1] + output_y_size;
    ioPointer[5] = ioBuffer[1] + output_y_size + output_uv_size;
  }

  requested_frames = std::vector<const VSFrame *> {size_t(config.batch_extract + int(config.extra_frame)), nullptr};
  return "";
}

std::string NNVISRFilter::init2(const VSFrame *frame, VSCore *core, const VSAPI *vsapi) {
  auto frame_prop = vsapi->getFramePropertiesRO(frame);
  first_frame = frame;
  int err;

  color_space_t def, cur;
  if (color_family == VSColorFamily::cfRGB) {
    def = {VSC_PRIMARIES_BT709, VSC_TRANSFER_BT709, VSC_MATRIX_RGB, VSC_RANGE_FULL};
  }
  else {
    def = {VSC_PRIMARIES_BT709, VSC_TRANSFER_BT709, VSC_MATRIX_BT709, VSC_RANGE_LIMITED};
  }

  cur.r = VSColorRange(vsapi->mapGetInt(frame_prop, "_ColorRange", 0, &err));
  if (err) {
    cur.r = def.r;
  }
  else if (cur.r > VSC_RANGE_LIMITED || cur.r < VSC_RANGE_FULL) {
    vsapi->logMessage(mtWarning, "NNVISR: input has invalid color range. Assuming default color range.", core);
    cur.r = def.r;
  }

  cur.cp = VSColorPrimaries(vsapi->mapGetInt(frame_prop, "_Primaries", 0, &err));
  if (err) {
    cur.cp = VSC_PRIMARIES_UNSPECIFIED;
  }
  switch (cur.cp) {
    case VSC_PRIMARIES_UNSPECIFIED:
      vsapi->logMessage(mtWarning, "NNVISR: input color primaries unspecified. Assuming default (BT.709).", core);
      cur.cp = def.cp;
      break;
    case VSC_PRIMARIES_ST240_M: cur.cp = VSC_PRIMARIES_ST170_M; break;
  }

  cur.tc = VSTransferCharacteristics(vsapi->mapGetInt(frame_prop, "_Transfer", 0, &err));
  if (err) {
    cur.tc = VSC_TRANSFER_UNSPECIFIED;
  }
  switch (cur.tc) {
    case VSC_TRANSFER_UNSPECIFIED:
      vsapi->logMessage(mtWarning, "NNVISR: input transfer characteristic unspecified. Assuming default (BT.709).",
                        core);
      cur.tc = def.tc;
      break;
    case VSC_TRANSFER_BT601:
    case VSC_TRANSFER_BT2020_10:
    case VSC_TRANSFER_BT2020_12: cur.tc = VSC_TRANSFER_BT709; break;
  }

  cur.mc = VSMatrixCoefficients(vsapi->mapGetInt(frame_prop, "_Matrix", 0, &err));
  if (err || cur.mc == VSC_MATRIX_UNSPECIFIED) {
    vsapi->logMessage(mtWarning, "NNVISR: input matrix coefficient unspecified. Assuming default (BT.709).", core);
    cur.mc = def.mc;
  }
  if (color_family == VSColorFamily::cfRGB && cur.mc != VSC_MATRIX_RGB) {
    vsapi->logMessage(mtWarning, "NNVISR: RGB input must uses RGB Matrix.", core);
    cur.mc = VSC_MATRIX_RGB;
  }
  else if (color_family == VSColorFamily::cfYUV) {
    switch (cur.mc) {
      case VSC_MATRIX_RGB:
        vsapi->logMessage(mtWarning, "NNVISR: YUV input must not use RGB Matrix, reset to default (BT.709).", core);
        cur.mc = def.mc;
      case VSC_MATRIX_BT470_BG: cur.mc = VSC_MATRIX_ST170_M; break;
      case VSC_MATRIX_CHROMATICITY_DERIVED_NCL:
        switch (cur.cp) {
          case VSC_PRIMARIES_BT709: cur.mc = VSC_MATRIX_BT709; break;
          case VSC_PRIMARIES_BT470_M: cur.mc = VSC_MATRIX_ST170_M; break;
          case VSC_PRIMARIES_BT2020: cur.mc = VSC_MATRIX_BT2020_NCL; break;
        }
    }
  }

  std::string colorspace_folder;
  if (vo.format.colorFamily == VSColorFamily::cfRGB) {
    std::stringstream ss;
    ss << "rgb_" << cur.cp << '_' << cur.tc;
    colorspace_folder = ss.str();
  }
  else {
    std::stringstream ss;
    ss << "yuv_" << cur.cp << '_' << cur.tc << '_' << cur.mc;
    colorspace_folder = ss.str();
  }

  ctx = new InferenceContext {config, *logger, engine_path / model / colorspace_folder};

  if (!ctx->has_file()) {
#if defined(_MSC_VER) && !defined(IS_CONDA_BUILD)
    // Dirty hack, but that's what you must pay if you want to fight with system default...

    // So here's the case:
    //  - By default Windows only search current *APPLICATION* dir for DLL dependencies, but not current DLL dir.
    //  - VapourSynth wants plugins and their dependencies to be packed together in the special "plugin dir",
    //    which is - fortunately and unfortunately - *NOT* the application dir. It tweaks `LoadLibrary` call to
    //    load the plugin correctly with non-default `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` flag.
    //  - However TensorRT and CUDA and cuDNN libraries are huge, and it's reasonable to lazy load them
    //    (and load each other) - which in turn loads dependencies, but with *DEFAULT* flag.
    //    And can't find them. Eww.
    //  - `AddDllDirectory` should generally be avoided, especially in libraries. But I really don't bother loading
    //    every library manually beforehand as it ruins the whole point of lazy loading.
    //    And I also don't bother instrumenting the `LoadLibrary` call.
    //    So either this directory is the autoloading directory, in which case it has the same effect of adding
    //    `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` to all `LoadLibrary` call, which VapourSynth is already doing anyway.
    //    Or there's only NNVISR and it's dependencies under this directory. If this introduce DLL conflict,
    //    then DLL conflict will happen anyway, because NNVISR will need them sooner or later.
    //  - And VapourSynth intentionally prevent plugins loading DLLs from PATH, unless with `altsearchpath=True`.
    //    Probably for good reason. For conda build we are circumventing this by lazy loading our dependencies as well,
    //    which are installed in PATH instead of plugin dir.
    //  - Hope there's no more issues.
    std::filesystem::path plugin_path = vsapi->getPluginPath(vsapi->getPluginByID("dev.tyty.aim.nnvisr", core));
    plugin_path.remove_filename();

    // We don't know if user used "altsearchpath" when calling `LoadPlugin`.
    // Guess from whether user placed dependencies at the plugin directory.
    if (exists(plugin_path / "nvinfer_builder_resource.dll")) {
      vsapi->logMessage(
          mtDebug, "NNVISR: dependencies under plugin path. Adding plugin directory to dll search directory.", core);
      SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      AddDllDirectory(std::filesystem::path(vsapi->getPluginPath(vsapi->getPluginByID("dev.tyty.aim.nnvisr", core)))
                          .remove_filename()
                          .c_str());
    }
#endif

    vsapi->logMessage(mtInformation, "NNVISR: building engine for current resolution. This will take some time.", core);
    OptimizationContext optimize_ctx {{config.input_width,
                                       config.input_height,
                                       {1, config.batch_extract, config.batch_extract},
                                       {1, config.batch_fusion, config.batch_fusion},
                                       config.input_count,
                                       config.feature_count,
                                       config.extraction_layers,
                                       config.extra_frame,
                                       config.double_frame,
                                       config.interpolation,
                                       config.scale_factor_w,
                                       config.scale_factor_h,
                                       config.format,
                                       config.use_fp16,
                                       config.low_mem},
                                      *logger,
                                      model_path,
                                      engine_path};
    err = optimize_ctx.optimize(model / colorspace_folder);
    if (err) {
      return "NNVISR: failed building engine for current input dimension";
    }
    vsapi->logMessage(mtInformation, "NNVISR: done building engine.", core);
  }

  if (!ctx->load_engine()) {
    delete ctx;
    delete logger;
    return "NNVISR: failed init context";
  }

  session = new InferenceSession {*ctx};
  if (!session->good()) {
    delete session;
    delete ctx;
    delete logger;
    return "NNVISR: failed init session";
  }

  if (raw_norm) {
    for (int i = 0; i < 3; ++i) {
      norm.z[i].a = 1.0f / denorm.z[i].a;
      norm.z[i].b = -denorm.z[i].b * norm.z[i].a;
    }
  }
  else {
    auto isFloat = vo.format.sampleType == VSSampleType::stFloat;
    auto depth = isFloat ? 8 : vo.format.bitsPerSample;
    if (isFloat && cur.r == VSColorRange::VSC_RANGE_LIMITED) {
      vsapi->logMessage(mtWarning,
                        "NNVISR: Normalization value for limited range floating point input may be inaccurate.", core);
    }

    if (color_family == VSColorFamily::cfYUV) {
      float kr, kg, kb;

      if (cur.mc == VSC_MATRIX_CHROMATICITY_DERIVED_NCL) {
        const float *pPrimaries = nullptr;
        for (const auto &entry: colorPrimariesTable) {
          if (cur.cp == entry.colorPrimariesEnum) {
            pPrimaries = entry.primaries;
            break;
          }
        }

        if (pPrimaries == nullptr) {
          vsapi->logMessage(mtWarning, "NNVISR: unknown color primary. Assume default (BT.709).", core);
          pPrimaries = colorPrimariesTable[0].primaries;
        }

        const auto [rX, rY, gX, gY, bX, bY, wX, wY] = (const float(&)[8])(*pPrimaries);
        float const rZ = 1.0f - (rX + rY);
        float const gZ = 1.0f - (gX + gY);
        float const bZ = 1.0f - (bX + bY);
        float const wZ = 1.0f - (wX + wY);
        kr = (rY * (wX * (gY * bZ - bY * gZ) + wY * (bX * gZ - gX * bZ) + wZ * (gX * bY - bX * gY))) /
             (wY * (rX * (gY * bZ - bY * gZ) + gX * (bY * rZ - rY * bZ) + bX * (rY * gZ - gY * rZ)));
        kb = (bY * (wX * (rY * gZ - gY * rZ) + wY * (gX * rZ - rX * gZ) + wZ * (rX * gY - gX * rY))) /
             (wY * (rX * (gY * bZ - bY * gZ) + gX * (bY * rZ - rY * bZ) + bX * (rY * gZ - gY * rZ)));
      }
      else {
        bool found = false;
        for (const auto &entry: matrixCoefficientsTable) {
          if (cur.mc == entry.matrixCoefficientsEnum) {
            kr = entry.kr;
            kb = entry.kb;
            found = true;
            break;
          }
        }

        if (!found) {
          return "NNVISR: unsupported matrix coefficient, can not infer normalization parameter.";
        }
      }

      kg = 1 - kr - kb;

      auto [rs, rm] = denorm.r;
      auto [gs, gm] = denorm.g;
      auto [bs, bm] = denorm.b;
      auto uv_bias = cur.r ? float(1 << (depth - 1)) / float((1 << depth) - 1) : 0.5f;

      denorm.y.b = rm * kr + gm * kg + bm * kb;
      denorm.y.a = std::sqrt(rs * rs * kr * kr + gs * gs * kg * kg + bs * bs * kb * kb);
      denorm.u.b = (bm - denorm.y.b) / (1 - kb) / 2 + uv_bias;
      denorm.u.a = std::sqrt(bs * bs + denorm.y.a * denorm.y.a) / (1 - kb) / 2;
      denorm.v.b = (rm - denorm.y.b) / (1 - kr) / 2 + uv_bias;
      denorm.v.a = std::sqrt(rs * rs + denorm.y.a * denorm.y.a) / (1 - kr) / 2;

      float uv_scale = std::sqrt(float(1 << vo.format.subSamplingW) * float(1 << vo.format.subSamplingH));
      denorm.u.a /= uv_scale;
      denorm.v.a /= uv_scale;
    }

    if (cur.r == VSC_RANGE_FULL) {
      y_max = uv_max = float((1 << depth) - 1);
      y_min = uv_min = 0;
    }
    else {
      y_max = float(235 << (depth - 8));
      uv_max = float(240 << (depth - 8));
      y_min = uv_min = float(16 << (depth - 8));
    }
    if (isFloat) {
      auto unorm = float((1 << depth) - 1);
      y_max /= unorm;
      y_min /= unorm;
      uv_max /= unorm;
      uv_min /= unorm;
    }

    for (int i = 0; i < 3; ++i) {
      float c = (color_family == VSColorFamily::cfRGB) ? (y_max - y_min) : (i == 0 ? y_max - y_min : uv_max - uv_min);
      float d = (color_family == VSColorFamily::cfRGB) ? (y_min) : (i == 0 ? y_min : uv_min);
      norm.z[i].a = 1.0f / denorm.z[i].a;
      norm.z[i].b = -(denorm.z[i].b + d / c) * norm.z[i].a;
      norm.z[i].a /= c;
      denorm.z[i].a *= c;
      denorm.z[i].b = denorm.z[i].b * c + d;
    }
  }

  return "";
}

template<class F, class U>
std::string NNVISRFilter::readPlane(md_view<F, 2> dst, md_uview<const U, 2> src, md_view<U, 2> cuda_tmp, float a,
                                    float b) {
  int err;
  if (src.is_contiguous()) {
    auto src_c = src.as_view();

    err = cudaMemcpyAsync(cuda_tmp.data, src_c.data, cuda_tmp.size() * sizeof(U), cudaMemcpyHostToDevice,
                          session->stream);
  }
  else {
    err = cudaMemcpy2DAsync(cuda_tmp.data, cuda_tmp.at(0).size() * sizeof(U), src.data, src.stride[0] * sizeof(U),
                            cuda_tmp.at(0).size() * sizeof(U), cuda_tmp.shape[0], cudaMemcpyHostToDevice,
                            session->stream);
  }

  if (err != cudaSuccess) {
    return "NNVISR: failed copy " + std::to_string(cuda_tmp.size() * sizeof(U)) +
           " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
           cudaGetErrorString(cudaError_t(err)) + ").";
  }

  import_pixel<F, U>(dst, cuda_tmp, a, b, session->stream);
  //  trace("put input " + describe(dst));
  return "";
}

template<class F, class U>
std::string NNVISRFilter::writePlane(md_uview<U, 2> dst, md_view<const F, 2> src, md_view<U, 2> cuda_tmp, float a,
                                     float b, float min, float max) {
  //  trace("get output " + describe(src));
  export_pixel<F, U>(cuda_tmp, src, a, b, min, max, session->stream);

  int err;
  if (dst.is_contiguous()) {
    auto dst_c = dst.as_view();

    err = cudaMemcpyAsync(dst_c.data, cuda_tmp.data, dst_c.size() * sizeof(U), cudaMemcpyDeviceToHost, session->stream);
  }
  else {
    err = cudaMemcpy2DAsync(dst.data, dst.stride[0] * sizeof(U), cuda_tmp.data, cuda_tmp.at(0).size() * sizeof(U),
                            cuda_tmp.at(0).size() * sizeof(U), cuda_tmp.shape[0], cudaMemcpyDeviceToHost,
                            session->stream);
  }

  if (err != cudaSuccess) {
    return "NNVISR: failed copy " + std::to_string(cuda_tmp.size() * sizeof(U)) +
           " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
           cudaGetErrorString(cudaError_t(err)) + ").";
  }

  return "";
}

template<class U>
std::string NNVISRFilter::uploadYUV(offset_t position, const VSFrame *frame, const VSAPI *vsapi) {
  auto bps = vo.format.bytesPerSample;
  md_uview<const U, 2> input_planes[3] = {
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 0)), input_shape_y, {vsapi->getStride(frame, 0) / bps, 1}},
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 1)), input_shape_uv, {vsapi->getStride(frame, 1) / bps, 1}},
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 2)),
       input_shape_uv,
       {vsapi->getStride(frame, 2) / bps, 1}}};
  md_view<U, 2> input_tmps[3] = {{reinterpret_cast<U *>(ioPointer[0]), input_shape_y},
                                 {reinterpret_cast<U *>(ioPointer[1]), input_shape_uv},
                                 {reinterpret_cast<U *>(ioPointer[2]), input_shape_uv}};

  for (int i = 0; i < 3; ++i) {
    shape_t<2> dim;
    uint8_t *tensor_ptr;
    if (i == 0) {
      dim = input_tensor_y;
      tensor_ptr = session->input.at(position, 0).data;
    }
    else {
      dim = input_tensor_uv;
      tensor_ptr = session->input_uv.at(position, i - 1).data;
    }

    std::string result;
    if (config.use_fp16) {
      result = readPlane<half, U>({(half *) tensor_ptr, dim}, input_planes[i], input_tmps[i], norm.z[i].a, norm.z[i].b);
    }
    else {
      result =
          readPlane<float, U>({(float *) tensor_ptr, dim}, input_planes[i], input_tmps[i], norm.z[i].a, norm.z[i].b);
    }
    if (!result.empty()) {
      return result;
    }
  }

  return "";
}

template<class U>
std::string NNVISRFilter::uploadRGB(offset_t position, const VSFrame *frame, const VSAPI *vsapi) {
  auto bps = vo.format.bytesPerSample;
  md_uview<const U, 2> input_planes[3] = {
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 0)), input_shape_y, {vsapi->getStride(frame, 0) / bps, 1}},
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 1)), input_shape_y, {vsapi->getStride(frame, 1) / bps, 1}},
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 2)), input_shape_y, {vsapi->getStride(frame, 2) / bps, 1}}};
  md_view<U, 2> input_tmps[3] = {{reinterpret_cast<U *>(ioPointer[0]), input_shape_y},
                                 {reinterpret_cast<U *>(ioPointer[1]), input_shape_y},
                                 {reinterpret_cast<U *>(ioPointer[2]), input_shape_y}};

  for (int i = 0; i < 3; ++i) {
    //    trace("input position " + std::to_string(position) + " writing to " + describe(session->input.at(position, i)));
    uint8_t *tensor_ptr = session->input.at(position, i).data;

    std::string result;
    if (config.use_fp16) {
      result = readPlane<half, U>({(half *) tensor_ptr, input_shape_y}, input_planes[i], input_tmps[i], norm.z[i].a,
                                  norm.z[i].b);
    }
    else {
      result = readPlane<float, U>({(float *) tensor_ptr, input_shape_y}, input_planes[i], input_tmps[i], norm.z[i].a,
                                   norm.z[i].b);
    }
    if (!result.empty()) {
      return result;
    }
  }

  return "";
}

template<class U>
std::string NNVISRFilter::downloadYUV(offset_t position, VSFrame *frame, const VSAPI *vsapi) {
  auto bps = vo.format.bytesPerSample;
  md_uview<U, 2> output_planes[3] = {
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 0)), output_shape_y, {vsapi->getStride(frame, 0) / bps, 1}},
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 1)), output_shape_uv, {vsapi->getStride(frame, 1) / bps, 1}},
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 2)), output_shape_uv, {vsapi->getStride(frame, 2) / bps, 1}}};
  md_view<U, 2> output_tmps[3] = {{reinterpret_cast<U *>(ioPointer[3]), output_shape_y},
                                  {reinterpret_cast<U *>(ioPointer[4]), output_shape_uv},
                                  {reinterpret_cast<U *>(ioPointer[5]), output_shape_uv}};

  auto idx = session->outputIndex(position);
  //  trace("output position " + std::to_string(position) + " is " + describe(idx));

  for (int i = 0; i < 3; ++i) {
    shape_t<2> dim;
    uint8_t *tensor_ptr;
    float min, max;
    if (i == 0) {
      dim = output_tensor_y;
      tensor_ptr = session->output.at(idx).at(0).data;
      min = y_min;
      max = y_max;
    }
    else {
      dim = output_tensor_uv;
      tensor_ptr = session->output_uv.at(idx).at(i - 1).data;
      min = uv_min;
      max = uv_max;
    }

    std::string result;
    if (config.use_fp16) {
      result = writePlane<half, U>(output_planes[i], {(half *) tensor_ptr, dim}, output_tmps[i], denorm.z[i].a,
                                   denorm.z[i].b, min, max);
    }
    else {
      result = writePlane<float, U>(output_planes[i], {(float *) tensor_ptr, dim}, output_tmps[i], denorm.z[i].a,
                                    denorm.z[i].b, min, max);
    }
    if (!result.empty()) {
      return result;
    }
  }

  return "";
}

template<class U>
std::string NNVISRFilter::downloadRGB(offset_t position, VSFrame *frame, const VSAPI *vsapi) {
  auto bps = vo.format.bytesPerSample;
  md_uview<U, 2> output_planes[3] = {
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 0)), output_shape_y, {vsapi->getStride(frame, 0) / bps, 1}},
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 1)), output_shape_y, {vsapi->getStride(frame, 1) / bps, 1}},
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 2)), output_shape_y, {vsapi->getStride(frame, 2) / bps, 1}}};
  md_view<U, 2> output_tmps[3] = {{reinterpret_cast<U *>(ioPointer[3]), output_shape_y},
                                  {reinterpret_cast<U *>(ioPointer[4]), output_shape_y},
                                  {reinterpret_cast<U *>(ioPointer[5]), output_shape_y}};

  auto idx = session->outputIndex(position);
  //  trace("output position " + std::to_string(position) + " is " + describe(idx));

  for (int i = 0; i < 3; ++i) {
    //    trace("output position " + std::to_string(position) + " reading from " + describe(session->output.at(idx).at(i)));
    uint8_t *tensor_ptr = session->output.at(idx).at(i).data;

    std::string result;
    if (config.use_fp16) {
      result = writePlane<half, U>(output_planes[i], {(half *) tensor_ptr, output_tensor_y}, output_tmps[i],
                                   denorm.z[i].a, denorm.z[i].b, y_min, y_max);
    }
    else {
      result = writePlane<float, U>(output_planes[i], {(float *) tensor_ptr, output_tensor_y}, output_tmps[i],
                                    denorm.z[i].a, denorm.z[i].b, y_min, y_max);
    }
    if (!result.empty()) {
      return result;
    }
  }

  return "";
}

std::string NNVISRFilter::requestFrames(int n, VSFrameContext *frameCtx, const VSAPI *vsapi) {
  //  auto request = [&](int i) { vsapi->requestFrameFilter(i, node, frameCtx); };
  auto request = [&](int i) {
    //        trace("requesting in frame #" + std::to_string(i));
    vsapi->requestFrameFilter(i, node, frameCtx);
  };

  if (last_output_frame + 1 != n) {
    return "NNVISR: unexpected request: non-linear frame request, requesting " + std::to_string(n) + " after " +
           std::to_string(last_output_frame);
  }
  last_output_frame = n;

  start_of_batch = false;
  start_of_scene = false;

  auto m = config.interpolation ? n / 2 : n;
  if (config.interpolation && n % 2 == 1) {
    request(last_requested_frame);
    return "";
  }

  if (config.extra_frame) {
    if (end_of_scene) {
      start_of_batch = m == extract_end;// open range
      start_of_scene = start_of_batch;
    }
    else {
      start_of_batch = extract_end != num_frames && m + 1 == extract_end;// one extra frame is consumed
      // [     )
      // {* *}
      // 0 1 2 3 4 5
      //     [     )
      //     {* *}
      // at m = 2, last batch consumed 0,1,2 so consumed_end = 3.
      // this batch will reuse 2 and consume 3,4
    }
  }
  else {
    start_of_batch = m == extract_end;
    start_of_scene = start_of_batch && end_of_scene;
  }
  end_of_scene = false;
  //  if (start_of_scene) {
  //    trace("Start of scene at out frame " + std::to_string(n));
  //  }

  if (!start_of_batch) {
    request(last_requested_frame);
    return "";
  }

  extract_begin = extract_end;
  //  trace("extract_begin at " + std::to_string(extract_begin));
  auto count = config.batch_extract;
  if (start_of_scene && config.extra_frame) {
    ++count;
  }
  extract_end = std::min(num_frames, extract_begin + count);
  //  trace("extract_end temporarily at " + std::to_string(extract_end));
  //  trace("SOB out frame " + std::to_string(n) + ", SOC: " + std::to_string(start_of_scene) + ", requesting " +
  //        std::to_string(extract_begin) + "-" + std::to_string(extract_end));
  if (extract_begin == extract_end) {
    // extra_frame mode, last frame was consumed by last batch
    assert(extract_begin != 0);
    --extract_begin;
    //    trace("extract_begin adjust to " + std::to_string(extract_begin));
  }
  for (int32_t i = extract_begin; i < extract_end; ++i) {
    request(i);
  }
  last_requested_frame = extract_end - 1;
  fusion_in_begin = extract_begin;

  return "";
}

std::string NNVISRFilter::prepareFrame(int n, VSFrameContext *frameCtx, const VSAPI *vsapi) {
  auto getFrame = [&, this](int k) {
    auto frame = k == 0 ? first_frame : vsapi->getFrameFilter(k, node, frameCtx);
    //        trace("acquire in frame #" + std::to_string(k));
    return frame;
  };
  auto loadFrameAt = [&](const VSFrame *frame_in, int32_t offset) -> std::string {
    if (color_family == VSColorFamily::cfRGB) {
      if (vi.format.sampleType == VSSampleType::stFloat) {
        if (vi.format.bytesPerSample == 2) {
          return uploadRGB<half>(offset, frame_in, vsapi);
        }
        else if (vi.format.bytesPerSample == 4) {
          return uploadRGB<float>(offset, frame_in, vsapi);
        }
      }
      else {
        if (vi.format.bytesPerSample == 1) {
          return uploadRGB<uint8_t>(offset, frame_in, vsapi);
        }
        else if (vi.format.bytesPerSample == 2) {
          return uploadRGB<uint16_t>(offset, frame_in, vsapi);
        }
      }
    }
    else {
      if (vi.format.sampleType == VSSampleType::stFloat) {
        if (vi.format.bytesPerSample == 2) {
          return uploadYUV<half>(offset, frame_in, vsapi);
        }
        else if (vi.format.bytesPerSample == 4) {
          return uploadYUV<float>(offset, frame_in, vsapi);
        }
      }
      else {
        if (vi.format.bytesPerSample == 1) {
          return uploadYUV<uint8_t>(offset, frame_in, vsapi);
        }
        else if (vi.format.bytesPerSample == 2) {
          return uploadYUV<uint16_t>(offset, frame_in, vsapi);
        }
      }

      return "NNVISR: unexpected format";
    }

    return "";
  };

  auto m = config.interpolation ? n / 2 : n;
  if (config.interpolation && n % 2 == 1) {
    return "";
  }

  if (start_of_batch) {
    int loaded_frames = 0;
    int recycle_frames = 0;
    // handle extra_frame
    if (config.extra_frame) {
      if (start_of_scene) {
        assert(m == extract_begin);
        requested_frames[0] = getFrame(extract_begin);
        ++loaded_frames;
        int err;// ignore key absent error: no scene change info is not critical
        if (vsapi->mapGetInt(vsapi->getFramePropertiesRO(requested_frames[0]), "_SceneChangePrev", 0, &err)) {
          end_of_scene = true;
          extract_end = extract_begin + 1;
          //          trace("extract_end at " + std::to_string(extract_end) + ", eos");
        }
      }
      else {
        assert(m + 1 == extract_begin);
        recycle_frames = 1;
      }
    }

    for (; loaded_frames < extract_end - extract_begin; ++loaded_frames) {
      auto frame_in = getFrame(extract_begin + loaded_frames);
      assert(frame_in);
      requested_frames[loaded_frames] = frame_in;
      int err;
      if (vsapi->mapGetInt(vsapi->getFramePropertiesRO(frame_in), "_SceneChangePrev", 0, &err)) {
        end_of_scene = true;
        extract_end = extract_begin + loaded_frames;
        //        trace("extract_end at " + std::to_string(extract_end) + ", eos");
        break;
      }
    }

    if (extract_end == num_frames) {
      end_of_scene = true;
    }

    // for extra_frame case the last frame of scene is virtually duplicated
    // so we can get exact 2x frames.
    auto least_frame_count = config.input_count;
    if (config.extra_frame && end_of_scene) {
      --least_frame_count;
    }
    if ((loaded_frames + recycle_frames) < least_frame_count && !start_of_scene) {
      recycle_frames = least_frame_count - loaded_frames;
    }

    if (recycle_frames) {
      std::rotate(requested_frames.rbegin(), requested_frames.rbegin() + recycle_frames, requested_frames.rend());
      auto recycle_begin = config.batch_extract - recycle_frames + 1;
      for (int i = 0; i < recycle_frames; ++i) {
        //        trace("recycle frame " + std::to_string(i) + " is placed at position " +
        //              std::to_string(session->internalFeatureIndex(i)));
        session->duplicateExtractOutput(recycle_begin + i, i);
      }
      extract_begin -= recycle_frames;// always the first frame number in feature buffer (extract output)
                                      //      trace("extract_begin adjust to " + std::to_string(extract_begin));
      fusion_in_begin = extract_begin;
    }

    // extra_frame, start_of_scene, full batch, so the first frame is extract separately
    if (start_of_scene && loaded_frames > config.batch_extract) {
      assert(recycle_frames == 0);
      session->extractBatch(0, 0, 1);
      //      trace("batch frame 0 is placed at position 0");
      loadFrameAt(requested_frames[0], 0);
      session->extract();
      --loaded_frames;
      ++recycle_frames;// from now on meaning of recycle_frames changed to number of frames already filled in feature buffer
    }

    if (recycle_frames > 1 || loaded_frames < config.batch_extract) {
      // we recycled frame from last batch, or don't have enough frame to make a batch,
      // which makes remain frames non-contiguous so can't be batched
      //      trace("non-full batch");
      for (int i = 0; i < loaded_frames; ++i) {
        loadFrameAt(requested_frames[recycle_frames + i], 0);
        //        trace("batch frame " + std::to_string(recycle_frames + i) + " is placed at position " +
        //              std::to_string(session->internalFeatureIndex(recycle_frames + i)));
        session->extractBatch(0, session->internalFeatureIndex(recycle_frames + i), 1);
        session->extract();
      }
    }
    else {
      for (int i = 0; i < loaded_frames; ++i) {
        //        trace("batch frame " + std::to_string(recycle_frames + i) + " is placed at position " +
        //              std::to_string(session->internalFeatureIndex(recycle_frames + i)));
        loadFrameAt(requested_frames[recycle_frames + i],
                    session->internalFeatureIndex(recycle_frames + i) - recycle_frames);
      }
      session->extractBatch(0, recycle_frames, config.batch_extract);
      session->extract();
    }
  }

  // doing fusion on extracted results.

  if (start_of_batch || m == fusion_in_begin) {
    // 1. try to consume n frame groups, with n <= config.batch_fusion
    //    If extra_frame and end_of_scene, consider last frame is duplicated
    // 2. adjust fusion_begin, try to work on latest config.input_count frame
    // 3. point last frames to duplicate frames to get config.input_count frame

    // e.g. for extra_frame case:
    // scene has 7 frames: 0 1 2 3 4 5 6
    // input_count = 4, fusion_batch = 1
    // fusion_begin = 0, extract_end = 7
    // 1. grouped fusion, batch = 1, fusion_begin = 3 now
    // 2. grouped fusion, batch = 1, fusion_begin = 6 now
    // *. fusion_begin + 1 == extract_end is done

    // e.g. for extra_frame and end_of_scene case:
    // scene has 2 frames: 0 1
    // input_count = 4
    // fusion_begin = 0, extract_end = 6, fusion_begin = 0
    // 1. custom fusion on 0 1 1 1, done

    // scene has 5 frames: 0 1 2 3 4
    // input_count = 4
    // fusion_begin = 0, extract_end = 6, fusion_begin = 0
    // 1. grouped fusion, batch = 1, fusion_begin = 3 now
    // 2. custom fusion on 2 3 4 4, done

    // scene has 6 frames: 0 1 2 3 4 5
    // input_count = 4
    // fusion_begin = 0, extract_end = 6, fusion_begin = 0
    // 1. grouped fusion, batch = 1, fusion_begin = 3 now
    // 2. custom fusion on 3 4 5 5, done

    // scene has 7 frames: 0 1 2 3 4 5 6
    // input_count = 4
    // fusion_begin = 0, extract_end = 7
    // 1. grouped fusion, batch = 2, fusion_begin = 6 now
    // 2. custom fusion on 4 5 6 6, done

    auto available = extract_end - fusion_in_begin - int(config.extra_frame);
    auto one_batch = config.input_count - int(config.extra_frame);

    auto batch = std::min(available / one_batch, config.batch_fusion);
    if (batch) {
      if (m != extract_begin) {
        auto fusion_idx = m - extract_begin;
        session->duplicateExtractOutput(fusion_idx, fusion_idx - 1);
      }

      auto offset = (fusion_in_begin - extract_begin) / (config.batch_fusion * one_batch);
      //      trace("batched fusion, processing " + std::to_string(batch) + " batch from " +
      //            std::to_string(offset * config.batch_fusion));
      session->fusionBatch(batch);
      session->fusionGroupedOffset(offset);
      session->fusion();
      fusion_out_begin = fusion_in_begin;
      fusion_in_begin += batch * one_batch;
    }
    else {
      auto end_idx = extract_end - 1;
      auto begin_idx = std::max(extract_begin, end_idx - one_batch + 1);
      std::vector<int32_t> indexes(config.input_count);
      for (int i = 0; i < config.input_count; ++i) {
        indexes[i] = std::min(begin_idx + i, end_idx) - extract_begin;
      }
      //      trace("non-batched fusion on frames " + std::to_string(begin_idx) + " to " + std::to_string(end_idx));
      session->fusionBatch(1);
      session->fusionCustomOffset(indexes);
      session->fusion();
      fusion_out_begin = begin_idx;
    }
  }

  return "";
}

std::string NNVISRFilter::extractFrame(int n, const VSFrame *&frame, VSCore *core, const VSAPI *vsapi) {
  int offset;
  int src_index;
  bool free_src;
  bool from_source;
  if (config.interpolation) {
    offset = n - 2 * fusion_out_begin;
    src_index = n / 2 - extract_begin;
    free_src = offset % 2;
    from_source = !config.double_frame && !free_src;
  }
  else {
    offset = n - fusion_out_begin;
    src_index = n - extract_begin;
    free_src = true;
    from_source = false;
  }

  auto adjust_duration = [&](VSFrame *frame, bool modify_timestamp) {
    auto prop = vsapi->getFramePropertiesRW(frame);
    auto num = vsapi->mapGetInt(prop, "_DurationNum", 0, nullptr) * 2;
    auto den = vsapi->mapGetInt(prop, "_DurationDen", 0, nullptr);
    vsh::reduceRational(&num, &den);
    vsapi->mapSetInt(prop, "_DurationNum", num, false);
    vsapi->mapSetInt(prop, "_DurationDen", den, false);
    if (modify_timestamp) {
      int err;
      auto time = vsapi->mapGetFloat(prop, "_AbsoluteTime", 0, &err);
      if (!err) {
        time += double(num) / den;
        vsapi->mapSetFloat(prop, "_AbsoluteTime", time, false);
      }
    }
  };

  if (from_source) {
    // model doing frame interpolation but not producing double frame,
    // so model only outputs intermediate frames (no enhancement for input frame)
    // in this case we just return original frame.
    auto n_frame = vsapi->copyFrame(requested_frames[src_index], core);
    if (config.interpolation) {
      adjust_duration(n_frame, false);
    }
    frame = n_frame;
    return "";
  }

  auto n_frame = vsapi->newVideoFrame(&vo.format, vo.width, vo.height, requested_frames[src_index], core);
  if (config.interpolation) {
    adjust_duration(n_frame, true);
  }
  frame = n_frame;
  if (free_src) {
    //    trace("free frame " + std::to_string(n / 2));
    vsapi->freeFrame(requested_frames[src_index]);
    requested_frames[src_index] = nullptr;
    if (end_of_scene && n + 1 == extract_end) {
      //      trace("free frame " + std::to_string(n / 2 + 1));
      vsapi->freeFrame(requested_frames[src_index + 1]);
      requested_frames[src_index + 1] = nullptr;
    }
  }

  if (color_family == VSColorFamily::cfRGB) {
    if (vo.format.sampleType == VSSampleType::stFloat) {
      if (vo.format.bytesPerSample == 2) {
        return downloadRGB<half>(offset, n_frame, vsapi);
      }
      else if (vo.format.bytesPerSample == 4) {
        return downloadRGB<float>(offset, n_frame, vsapi);
      }
    }
    else {
      if (vo.format.bytesPerSample == 1) {
        return downloadRGB<uint8_t>(offset, n_frame, vsapi);
      }
      else if (vo.format.bytesPerSample == 2) {
        return downloadRGB<uint16_t>(offset, n_frame, vsapi);
      }
    }
  }
  else {
    if (vo.format.sampleType == VSSampleType::stFloat) {
      if (vo.format.bytesPerSample == 2) {
        return downloadYUV<half>(offset, n_frame, vsapi);
      }
      else if (vo.format.bytesPerSample == 4) {
        return downloadYUV<float>(offset, n_frame, vsapi);
      }
    }
    else {
      if (vo.format.bytesPerSample == 1) {
        return downloadYUV<uint8_t>(offset, n_frame, vsapi);
      }
      else if (vo.format.bytesPerSample == 2) {
        return downloadYUV<uint16_t>(offset, n_frame, vsapi);
      }
    }

    return "NNVISR: unexpected format";
  }

  return "";
}

NNVISRFilter::~NNVISRFilter() {
  for (auto p: ioBuffer) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// VS API

static const VSFrame *VS_CC NNVISRGetFrame(int n, int activationReason, void *instanceData, void **frameData,
                                           VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
  auto *filter = static_cast<NNVISRFilter *>(instanceData);
  std::string err;

  if (activationReason == arInitial) {
    err = filter->requestFrames(n, frameCtx, vsapi);
    if (!err.empty()) {
      vsapi->setFilterError(err.c_str(), frameCtx);
      vsapi->freeNode(filter->node);
    }
    return nullptr;
  }
  else if (activationReason == arAllFramesReady) {
    if (n == 0) {
      const VSFrame *frame = vsapi->getFrameFilter(0, filter->node, frameCtx);
      err = filter->init2(frame, core, vsapi);
      if (!err.empty()) {
        vsapi->setFilterError(err.c_str(), frameCtx);
        vsapi->freeNode(filter->node);
        return nullptr;
      }
    }

    err = filter->prepareFrame(n, frameCtx, vsapi);
    if (!err.empty()) {
      vsapi->setFilterError(err.c_str(), frameCtx);
      vsapi->freeNode(filter->node);
      return nullptr;
    }

    const VSFrame *out {};
    err = filter->extractFrame(n, out, core, vsapi);
    if (err.empty()) {
      err = filter->synchronize();
    }
    if (!err.empty()) {
      vsapi->setFilterError(err.c_str(), frameCtx);
      vsapi->freeNode(filter->node);
      return nullptr;
    }
    return out;
  }
  return nullptr;
}

static void VS_CC NNVISRFree(void *instanceData, VSCore *, const VSAPI *vsapi) {
  auto *d = static_cast<NNVISRFilter *>(instanceData);
  vsapi->freeNode(d->node);
  for (auto f: d->requested_frames) {
    if (f) {
      vsapi->freeFrame(f);
    }
  }
  delete d;
}

static void VS_CC NNVISRCreate(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *vsapi) {
  auto filter = new NNVISRFilter();
  auto err = filter->init1(in, core, vsapi);
  if (!err.empty()) {
    vsapi->mapSetError(out, err.c_str());
    vsapi->freeNode(filter->node);
    return;
  }

  VSFilterDependency deps[] = {{filter->node, rpNoFrameReuse}};
  vsapi->createVideoFilter(out, "NNVISR", &filter->vo, NNVISRGetFrame, NNVISRFree, fmFrameState, deps, 1, filter, core);
  auto out_node = vsapi->mapGetNode(out, "clip", 0, nullptr);
  vsapi->setLinearFilter(out_node);
  vsapi->freeNode(out_node);
}

static void VS_CC dependencyVersion(const VSMap *, VSMap *out, void *, VSCore *, const VSAPI *vsapi) {
  vsapi->mapSetData(out, "tensorrt_version", std::to_string(getInferLibVersion()).c_str(), -1, ptData, maReplace);

  vsapi->mapSetData(out, "tensorrt_version_build", std::to_string(NV_TENSORRT_VERSION).c_str(), -1, ptData, maReplace);

  int runtime_version;
  cudaRuntimeGetVersion(&runtime_version);
  vsapi->mapSetData(out, "cuda_runtime_version", std::to_string(runtime_version).c_str(), -1, ptData, maReplace);

  vsapi->mapSetData(out, "cuda_runtime_version_build", std::to_string(__CUDART_API_VERSION).c_str(), -1, ptData,
                    maReplace);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
  UDOLayers::registerPlugins();
  vspapi->configPlugin("dev.tyty.aim.nnvisr", "nnvisr", "Neural Network Video Interpolation / Super Resolution Filter",
                       VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
  vspapi->registerFunction("Super",
                           "clip:vnode;"
                           "scale_factor:float;"
                           "scale_factor_h:float:opt;"
                           "batch_size_extract:int:opt;"
                           "batch_size_fusion:int:opt;"
                           "input_count:int:opt;"
                           "feature_count:int:opt;"
                           "extraction_layers:int:opt;"
                           "interpolation:int:opt;"
                           "extra_frame:int:opt;"
                           "double_frame:int:opt;"
                           "use_fp16:int:opt;"
                           "norm_mean:float[]:opt;"
                           "norm_std:float[]:opt;"
                           "raw_norm:int:opt;"
                           "model:data:opt;"
                           "model_path:data:opt;"
                           "engine_path:data:opt;"
                           "low_mem:int:opt;",
                           "clip:vnode;", NNVISRCreate, nullptr, plugin);
  vspapi->registerFunction("Version", "", "", dependencyVersion, nullptr, plugin);
}
