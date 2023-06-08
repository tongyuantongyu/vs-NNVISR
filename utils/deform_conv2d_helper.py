import torch.onnx

@torch.onnx.symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def symbolic_deform_conv2d_forward(g,
                                   input,
                                   weight,
                                   offset,
                                   mask,
                                   bias,
                                   stride_h,
                                   stride_w,
                                   pad_h,
                                   pad_w,
                                   dil_h,
                                   dil_w,
                                   n_weight_grps,
                                   n_offset_grps,
                                   use_mask):
    if n_weight_grps != 1 or not use_mask:
        raise NotImplementedError()
    return g.op("custom::DeformConv2d", input, offset, mask, weight, bias, stride_i=[stride_h, stride_w],
                padding_i=[pad_h, pad_w], dilation_i=[dil_h, dil_w], deformable_groups_i=n_offset_grps,
                activation_type_i=-1, alpha_f=0.0, beta_f=0.0)


# Register custom symbolic function
torch.onnx.register_custom_op_symbolic("torchvision::deform_conv2d", symbolic_deform_conv2d_forward, 1)
