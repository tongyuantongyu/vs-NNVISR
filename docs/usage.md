# Usage

NNVISR works like any VapourSynth plugin. You may take a look at
[VapourSynth documentation](http://vapoursynth.com/doc/introduction.html)
if you are not familiar with it.
It registers the `nnvisr` namespace, and provide a single Function `Super`
to perform enhancement.

## Function interface

The required parameters of `Super` are list below. Parameters denoted with (*)
at end are determined by the network you are using, and should be provided
along with the model file.

- `clip`: The input clip. It is required to be in RGB or YUV420 format, and has
  constant frame rate. It's highly recommended to attach scene change properties 
  to `clip` before applying NNVISR.
- `scale_factor`: float. The scale factor of the network in use. (*)

The optional parameters of `Super` are:
- `scale_factor_h`: float. The scale factor of height of the network in use.
  This is default tobe the same as `scale_factor`. (*)
- `batch_size_extract`: int. The batch size of Extract model. Default automatically
  selected depending on other parameters.
- `batch_size_extract`: int. The batch size of Fusion model. Default to 1.
- `input_count`: int. The number of input frames network needed. Default to 1. (*)
- `feature_count`: int. The "feature" (`C` channel) size. Default to 64. (*)
- `extraction_layers`: int. The number of layers Extract model outputs. Default to 1. (*)
- `interpolation`: bool. If the network is doing frame interpolation
  (i.e. output clip) will have double framerate Default to False.
- `extra_frame`: bool. If network need 1 more input frame than consumed.
  Default to False. (*)
- `double_frame`: bool. If network outputs 2 times of frames than input.
  Default to False. (*)
- `use_fp16`: bool. Use half precision during inference. On supported GPUs
  (starting from Volta), this is usually ~2x faster and consumes half
  amount of GPU memory, but may cause numeric instability for some
  networks.
- `low_mem`: bool. Tweak TensorRT configurations to reduce memory usage.
  May cause performance degradation and effectiveness varies depending on
  actual model. Default to False.
- `norm_mean` and `norm_std`: float[3]. Normalization mean and std applied
  to inputs and output. The interpretation of these values depending on the
  following option. Defaults to [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] (*)
- `raw_norm`; bool. If True, `norm_mean` and `norm_std` are applied directly
  to the input and output frame pixel value of each channel.
  If False, `norm_mean` and `norm_std` are values of RGB channels from
  0-1 range. The actual value used for normalization is inferred automatically
  from colorspace information of input clip. Default to False. (*)
- `model`: str. The name of the model to be used. Default to ".". (*)
- `model_path`: str. The path that stores the model files.
  Default to `dev.tyty.aim.nnvisr` folder under the folder of plugin
  DLL.

## Adding network to NNVISR

The model files you got should be a folder. Place the whole folder
under `{model_path}/models` to allow NNVISR to find it. To use the network,
set `model` to the name of the folder you get when calling NNVISR.

## Notes

As the nature of TensorRT, NNVISR need some time to profile and build
execution engines for each model and each input resolution.
The built engines and profiling caches is also stored in `model_path`.

Video interpolation process is intrinsically sequential, so you should
only make one concurrent frame request when using NNVISR.
For `vspipe`, add `-r 1` to the parameters. NNVISR will make concurrent
frame request to upstream filters if you set a large enough batch size.
NNVISR is usually the bottleneck of your whole processing script so
mutlithreading downstream filters usually won't make processing faster.
