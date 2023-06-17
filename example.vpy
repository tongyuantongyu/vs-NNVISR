import vapoursynth as vs

core = vs.core

# Load example.mp4 as input video
# The `bestsource` plugin can be installed by running
# conda install -c conda-forge -c tongyuantongyu vapoursynth-bestsource
clip = core.bs.VideoSource(source="example.mp4")

# Using cycmunet/vimeo90k-deblur model.
# The config is copied from the corresponding table in
# https://github.com/tongyuantongyu/vs-NNVISR/blob/main/docs/models.md
model_config = {'scale_factor': 2, 'input_count': 2, 'feature_count': 64, 'extraction_layers': 4, 'interpolation': True, 'extra_frame': True, 'double_frame': True, 'model': 'cycmunet/vimeo90k-deblur'}

# cycmunet/vimeo90k-deblur only has model file handling YUV420 input.
# If your input is not YUV420 then it should be converted.
# For RGB models, use format=vs.RGB24 instead
# clip = clip.resize.Bilinear(format=vs.YUV420P8)

clip = core.nnvisr.Super(clip,
                         model_path=r"./model_path",  # update to where you place your model files
                         **model_config)

# Set as output
clip.set_output()

# To encode as video, run the following command
# You can use any other encoding tools that supports reading Y4M from stdin.
# vspipe example.vpy -c y4m -r 1 - | ffmpeg -i - output.mp4
