# Model Integration

NNVISR supports loading any model following a general interface defined below.
Model files integrated by us are available at
[Models](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/models.md).

## Model requirement

NNVISR supports neural network models working on a definite number of
consecutive frames.

Define `n` as the frame your network "consumes" for each run, or,
the difference of indexes of the first input frame between two consecutive inference.
Your network should take either `n` or `n + 1`
consecutive frames as input, and `n` or `2 n` frames as output.
If network outputs `n` frames, it can be treated as either frames
at the same timestamp of input frames (therefore replaced input frames)
or frames at intermediate timestamp (therefore interleaved with input frames).
If network outputs `2 n` frames, it always replaces input frames with output ones.

The network should be separated into 2 parts: Extract and Fusion.
Extract part should work on individual frame to produce either one
feature tensor, or a pyramid of multiscale feature tensors.
Fusion part should then take the features of consecutive `n` or `n + 1` frames
as input and outputs `n` or `2 n` frames.

Note that you can always let Extract part be an identity model. But if it's
possible, separating frame-independent part out can reduce repetitive work
and achieve higher performance.

Model can accept input and produce output in either RGB or YUV420 format
in NCHW layout.
For RGB I/O, `C` will be 3. For YUV420 I/O, The model should accept
two input tensors for each frame: the Y input `C` will be 1; the UV input
`C` will be 2, and `HW` will be ceiling half of Y input's `HW`. Output works similarly.

## Model format

NNVISR accepts neural network models in ONNX format.

### Naming for Input & Output

NNVISR uses a systematic naming to recognize each input and output.
Your provided model file should follow this naming to be correctly
loaded by NNVISR.

Define `m` as the number of layers Extract model produces.

#### Extract model

Extract model should have 1 input named `rgb` for RGB frame input,
or 2 inputs each named `y` and `uv` for YUV420 frame input.

Extract model should have `m` outputs, each named `l{i}`. For example,
if network produces 3 layers of pyramid, then model outputs should be
named `l0`, `l1`, `l2`. `l0` is the output with the same `HW` as input,
`l1`'s `HW` is the ceiling half of `l0`, and so on.

#### Fusion model

Depending on whether your network needs an extra input frame, Fusion model
can have either `m * n` or `m * (n + 1)` inputs. 

If your network is doing interpolation (output clip has double framerate),
Fusion model input should each named `f{2 * j}l{i}`, where `f{2 * j}l{i}`
refers to the `i`-th layer (`l{i}`) of `j`-th input frame.

If your network is not doing interpolation,
Fusion model input should each named `f{j}l{i}`, with similar meaning.

Fusion model with RGB outputs should have `n` or `2 n` outputs,
each named `h{k}`, where `h{k}` refers to the
`k`-th frame in the output sequence.

Fusion model with YUV420 outputs should have twice outputs of 
corresponding RGB-output model, each named `h{k}_y` or `h{k}_uv`, refering
to the `y` or `uv` channels of `k`-th frame in the output sequence.

For example, for a RGB model whose `n` is 2:
- If your network outputs `n` frames that
replace input frames, then they should be named `h0`, `h1` and `h2`,
where `h0` is the first output frame and so on.
- If your network outputs `n` frames that interleave with input frames,
then they should be named `h1`, `h3` and `h5`, where `h1` is the first
output frame and is right after the first input frame, and so on.
- If your network outputs `2 n` frames, then they should be named
`h0`, `h1`, `h2`, `h3`, `h4` and `h5`, where `h0` is the first output frame
and so on.

### Dynamic Axes and Dimensions

NNVISR requires the N(batch), H(height) and W(width) axes of inputs and outputs
to be dynamic, in order to support input clips of any resolution,
and allow users to configure batching for better performance.

For UV channels of YUV420 I/O, and pyramid features I/O tensors that have axes
that are half of some value, "half" refers to ceiling half, or (x + 1) / 2,
instead of the usual flooring half.
This is to ensure the borders are kept instead of trimmed out.
You should take extra care of this when designing your network,
or document clearly the minimal alignment requirement of your network
so that users can insert suitable `AddBorders` and `Crop` filters
to run your network correctly.

For non-integer scale ratio, the output dimension should be the flooring
integers of input dimension multiplies scale factor.

## Non-standard Network operators

TensorRT may not support some special neural network operators.

Specially, NNVISR provides the Deformable Convolution (DCNv2) implementation
for TensorRT, so networks using DCNs are supported by NNVISR without further
work. For NNVISR to correctly recognize DCN operator in your model, use
namespace `custom` and name `DeformConv2d`.

The inputs of `DeformConv2d` in ONNX should be provided in order
`input`, `offset`, `mask`, `weight`, `bias`.

The attributes of `DeformConv2d` are:

| name              | type  | description                                                                                                     |
|-------------------|-------|-----------------------------------------------------------------------------------------------------------------|
| deformable_groups | int   | The deformable groups, or sometimes called offset groups.                                                       |
| dilation          | int[] | Two numbers represent the dilation of H and W. (note: H before W)                                               |
| padding           | int[] | Two numbers represent the padding of H and W. (note: H before W)                                                |
| stride            | int[] | Two numbers represent the stride of H and W. (note: H before W)                                                 |
| activation_type   | int   | The type of activation applied to outputs. Currently only -1 (no activation) and 3 (Leaky ReLU) is implemented. |
| alpha             | float | Alpha parameter for activation, if applicable.                                                                  |
| beta              | float | Beta parameter for activation, if applicable.                                                                   |

For PyTorch, we provide a helper script
[`utils/deform_conv2d_helper.py`](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/utils/deform_conv2d_helper.py),
which configures PyTorch to automatically exports `deform_conv2d` from
`torchvision` to `DeformConv2d` operator we recognize when exporting ONNX.

## Model folder and file

To let your model loaded by NNVISR it should be placed in correct place and
named correctly.

### Model folder structure

All model files should be put into a folder with name of your choice (can also have multiple levels).
This name is used as the `model` argument when calling NNVISR filter.

Inside it should be a number of folders to specify the I/O frame
characteristics of your network, which contains the actual model files in ONNX
format. For networks with RGB I/O, folder name
should be `rgb_{primary}_{transfer}`;
for networks with YUV I/O, folder name should be
`yuv_{primary}_{transfer}_{matrix}`.
`primary`, `tranfer` and `matrix` are numbers following
[definition of VapourSynth Resize filter](http://vapoursynth.com/doc/functions/video/resize.html)'s
parameter of the same name.
These numbers should be determined based on your dataset.
If you are not sure, they are usually `primary=1`, `transfer=1` and `matrix=6`.
However, we recommend train YUV network with configurations of `matrix=1`,
since it's more widely used in videos.

Refer to [Models](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/models.md) page
for examples to arrange model files.

### Model name

There should be two model files each for the Extract and Fusion part of your
network. Extract model should be named
`fe_n{input_count}_{scale_factor}x{scale_factor_h}_l{extraction_layers}.onnx`,
and Fusion model named
`ff_n{input_count}_{scale_factor}x{scale_factor_h}_l{extraction_layers}.onnx`.
For example, in a network accepting 3 frames for each run, scales width to 4x
and height to 2x, and produces 3 extraction layers, the Extract model should
be named `fe_n3_4x2_l3.onnx` and Fusion named `ff_n3_4x2_l3.onnx`.
For network with YUV420 I/O, append `_yuv1-1` to the name before extension.
The network with same configuration should then be named
`fe_n3_4x2_l3_yuv1-1.onnx` and `ff_n3_4x2_l3_yuv1-1.onnx` respectively.
