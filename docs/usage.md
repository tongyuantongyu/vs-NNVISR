# Usage

NNVISR works like any VapourSynth plugin. You may take a look at
[VapourSynth documentation](http://vapoursynth.com/doc/introduction.html)
if you are not familiar with it.
NNVISR uses `nnvisr` namespace, and provide a single Entrypoint `Super`
to perform enhancement.

[`example.vpy`](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/example.vpy)
provides a simple example for how to use NNVISR.

## Function interface

Required parameters of `Super` are list below. Parameters marked with (*)
are determined by the network you are using, and should be provided
along with the model file.

- `clip`: The input clip. It is required to be in RGB or YUV420 format, and has
  constant frame rate. It's highly recommended to attach scene change properties
  (`_SceneChangePrev`) to `clip` before applying NNVISR.
- `scale_factor`(*): float. The scale factor of the network in use.

Optional parameters of `Super` are:
- `scale_factor_h`(*): float. Height scale factor of the network.
  Default value: `scale_factor`.
- `batch_size_extract`: int. Batch size of Extract model. Default is automatically
  selected based on other parameters.
- `batch_size_fusion`: int. Batch size of Fusion model. Default value: 1.
- `input_count`(*): int. The number of input frames network needed. Default value: 1.
- `feature_count`(*): int. "feature" (`C` channel) axis size of extraction output and fusion input.
  Default value: 64.
- `extraction_layers`(*): int. The number of layers Extract model outputs. Default: 1.
- `interpolation`(*): bool. If the network is doing frame interpolation
  (i.e. output clip) will have double framerate. Default: False.
- `extra_frame`(*): bool. If network need 1 more input frame than consumed.
  Default: False.
- `double_frame`(*): bool. If network outputs 2 times of frames than input.
  Default: False.
- `use_fp16`: bool. Use half precision during inference. On supported GPUs
  (starting from Volta), this is usually ~2x faster and consumes half
  amount of GPU memory, but may cause numeric instability for some
  models. Default: False.
- `low_mem`: bool. Tweak TensorRT configurations to reduce memory usage.
  May cause performance degradation and effectiveness varies depending on
  actual model. Default: False.
- `norm_mean` and `norm_std`(*): float[3]. Normalization mean and std applied
  to inputs and output. The interpretation of these values depending on the
  following option. Default: [0, 0, 0] and [1, 1, 1]. 
- `raw_norm`(*): bool. If True, `norm_mean` and `norm_std` are applied directly
  to the input and output frame pixel value of each channel.
  If False, `norm_mean` and `norm_std` are values of RGB channels from
  0-1 range. The actual value used for normalization is inferred automatically
  from colorspace information of input clip. Default: False.
- `model`(*): str. The name of the model to be used. Default: ".".
- `model_path`: str. The path that stores the model files.
  Default: `dev.tyty.aim.nnvisr` folder alongside the plugin DLL.
- `engine_path`: str. Persistent cache dir to store TensorRT engine files.
  Default: a path under `model_path` unique to current GPU and TensorRT version.

## Adding network to NNVISR

Model files you downloaded should be a folder. Place the whole folder
under `{model_path}/models` to allow NNVISR to find it. To use the network,
set `model` to the name of the folder you get when calling NNVISR.
Model files, and detail on how to let NNVISR find them can be found at
[Models](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/models.md).

## Notes

Due to the way TensorRT works, NNVISR requires some time to profile and build
execution engines for each model and each input resolution.
The built engines and profiling caches is stored in `model_path`.

Video interpolation process is intrinsically sequential, so you should
always linearly iterate through output when using NNVISR.
NNVISR will error when out of order frame request is detected.
For `vspipe`, add `-r 1` to the parameters to ensure it.
NNVISR will make concurrent frame request to upstream filters
if you set a large enough batch size.
NNVISR is usually the bottleneck of the whole processing pipeline, so
mutlithreading downstream filters usually won't make processing faster.
