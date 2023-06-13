# Models

Here we are some model files for NNVISR.

To use, download from the link and put unzipped folder under `{model_path}/models`,
and call `nnvisr.Super` with [function parameters marked with (*)](https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/usage.md#function-interface)
using values given under "Config" column.

## CycMuNet+

Article:
[Spatial-Temporal Space Hand-in-Hand: Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Spatial-Temporal_Space_Hand-in-Hand_Spatial-Temporal_Video_Super-Resolution_via_Cycle-Projected_Mutual_Learning_CVPR_2022_paper.html)

| SR | Interpolation | Format | CP/TC/MC | Download                                                                                                                                   | Config                                                                                                                                                                                          | Note                                                         |
|----|---------------|--------|----------|--------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| 2x | ✅             | YUV420 | 1/1/1    | [Link](https://whueducn-my.sharepoint.com/:u:/g/personal/2018302110332_whu_edu_cn/EUjQDJUxR6hCoeCQcWE9a94Bngj5bgewwI2-NMqL5DZWPQ?e=F8bVTn) | ```{'scale_factor': 2, 'input_count': 2, 'feature_count': 64, 'extraction_layers': 4, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'cycmunet/vimeo90k-deblur'}``` | Trained on Vimeo90k triplet dataset with random blur applied |

## YOGO

Article:
[You Only Align Once: Bidirectional Interaction for Spatial-Temporal Video Super-Resolution](https://dl.acm.org/doi/abs/10.1145/3503161.3547874)

| SR | Interpolation | Format | CP/TC/MC | Download                                                                                                                                   | Config                                                                                                                                                                                   | Note                                  |
|----|---------------|--------|----------|--------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| 4x | ✅             | RGB    | 1/1/-    | [Link](https://whueducn-my.sharepoint.com/:u:/g/personal/2018302110332_whu_edu_cn/EYe1BWWpN1FGgmHnlb5GHNwBDREpAMFPI-CW8I9KqZBFQQ?e=qEFaic) | ```{'scale_factor': 4, 'input_count': 4, 'feature_count': 64, 'extraction_layers': 1, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'cycmunet/vimeo90k'}``` | Trained on Vimeo90k suptuplet dataset |
