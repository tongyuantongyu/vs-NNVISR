# Models

This is a list of models for NNVISR made by us.
To integrate new model to NNVISR, see [Integration Document](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/integration.md).

To use these models, download from the link, unzip, and put unzipped folder under `{model_path}/models`.
When calling `nnvisr.Super`, [function parameters marked with (*)](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/usage.md#function-interface)
should use values provided under "Config" column.

After unzipped, your `model_path` should look like this:

```
model_path
└── models
    ├── cycmunet
    │   └── vimeo90k-deblur
    │       └── yuv_1_1_1
    │           ├── fe_n2_2x2_l4_yuv1-1.onnx
    │           └── ff_n2_2x2_l4_yuv1-1.onnx
    └── (... more models ...)
```

For a manual installation with default `model_path`,
your VapoursSynth plugin folder should look like this:
```
vs-plugins
├── vs-nnvisr.dll
└── dev.tyty.aim.nnvisr
    └── models
        ├── cycmunet
        │   └── vimeo90k-deblur
        │       └── yuv_1_1_1
        │           ├── fe_n2_2x2_l4_yuv1-1.onnx
        │           └── ff_n2_2x2_l4_yuv1-1.onnx
        └── (... more models ...)

```

## Zooming Slow-Mo

Paper:
[Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution](https://arxiv.org/abs/2002.11616)

| SR | Input Frames | Interpolation | Format | CP/TC/MC | Download                                                                                               | Config                                                                                                                                                                              | Note                                                              |
|----|--------------|---------------|--------|----------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| 4x | 2            | ✅             | RGB    | 1/1/-    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/zooming-slowmo-2frame.zip) | ```{'scale_factor': 4, 'input_count': 2, 'feature_count': 64, 'extraction_layers': 3, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'zsm/vimeo90k'}``` | Official pretrained weights trained on Vimeo90k septuplet dataset |
| 4x | 4            | ✅             | RGB    | 1/1/-    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/zooming-slowmo-4frame.zip) | ```{'scale_factor': 4, 'input_count': 4, 'feature_count': 64, 'extraction_layers': 3, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'zsm/vimeo90k'}``` | Official pretrained weights trained on Vimeo90k septuplet dataset |

## TMNet

Paper:
[Temporal Modulation Network for Controllable Space-Time Video Super-Resolution](https://arxiv.org/abs/2104.10642)

| SR | Input Frames | Interpolation | Format | CP/TC/MC | Download                                                                                               | Config                                                                                                                                                                                | Note                                                              |
|----|--------------|---------------|--------|----------|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| 4x | 2            | ✅             | RGB    | 1/1/-    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/zooming-slowmo-2frame.zip) | ```{'scale_factor': 4, 'input_count': 2, 'feature_count': 64, 'extraction_layers': 3, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'tmnet/vimeo90k'}``` | Official pretrained weights trained on Vimeo90k septuplet dataset |
| 4x | 4            | ✅             | RGB    | 1/1/-    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/zooming-slowmo-4frame.zip) | ```{'scale_factor': 4, 'input_count': 4, 'feature_count': 64, 'extraction_layers': 3, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'tmnet/vimeo90k'}``` | Official pretrained weights trained on Vimeo90k septuplet dataset |


## CycMuNet+

Paper:
[Spatial-Temporal Space Hand-in-Hand: Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning](https://arxiv.org/abs/2205.05264)

| SR | Input Frames | Interpolation | Format | CP/TC/MC | Download                                                                                     | Config                                                                                                                                                                                                                                                                 | Note                                                         |
|----|--------------|---------------|--------|----------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| 2x | 2            | ✅             | YUV420 | 1/1/1    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/cycmunet-2x.zip) | ```{'scale_factor': 2, 'input_count': 2, 'feature_count': 64, 'extraction_layers': 4, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'cycmunet/vimeo90k-deblur', 'norm_mean': [0.485, 0.456, 0.406], 'norm_std': [0.229, 0.224, 0.225]}``` | Trained on Vimeo90k triplet dataset with random blur applied |

## VideoINR

Note: This model is only supported by TensorRT 8.6 version of NNVISR.
This model does not work correctly with `use_fp16=True`.

Paper:
[VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution](https://arxiv.org/abs/2206.04647)

| SR | Input Frames | Interpolation | Format | CP/TC/MC | Download                                                                                 | Config                                                                                                                                                                              | Note                                                    |
|----|--------------|---------------|--------|----------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| 2x | 2            | ✅             | RGB    | 1/1/-    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/vinr-2x.zip) | ```{'scale_factor': 2, 'input_count': 2, 'feature_count': 3, 'extraction_layers': 1, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'vinr/adobe240'}``` | Official pretrained weights trained on Adobe240 dataset |
| 4x | 2            | ✅             | RGB    | 1/1/-    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/vinr-4x.zip) | ```{'scale_factor': 4, 'input_count': 2, 'feature_count': 3, 'extraction_layers': 1, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'vinr/adobe240'}``` | Official pretrained weights trained on Adobe240 dataset |


## YOGO

Paper:
[You Only Align Once: Bidirectional Interaction for Spatial-Temporal Video Super-Resolution](https://arxiv.org/abs/2207.06345)

| SR | Input Frames | Interpolation | Format | CP/TC/MC | Download                                                                                     | Config                                                                                                                                                                               | Note                                  |
|----|--------------|---------------|--------|----------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| 4x | 2            | ✅             | RGB    | 1/1/-    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/yogo-2frame.zip) | ```{'scale_factor': 4, 'input_count': 2, 'feature_count': 64, 'extraction_layers': 1, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'yogo/vimeo90k'}``` | Trained on Vimeo90k septuplet dataset |
| 4x | 4            | ✅             | RGB    | 1/1/-    | [Link](https://github.com/tongyuantongyu/vs-NNVISR/releases/download/assets/yogo-4frame.zip) | ```{'scale_factor': 4, 'input_count': 4, 'feature_count': 64, 'extraction_layers': 1, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'yogo/vimeo90k'}``` | Trained on Vimeo90k septuplet dataset |
