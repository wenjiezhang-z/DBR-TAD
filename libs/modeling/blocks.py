import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .weight_init import trunc_normal_
import math
from functools import partial
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.layers.helpers import to_2tuple

def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T // self.stride,
                mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            expansion_factor=2,  # expansion factor of feat dims
            n_out=None,  # output dimension, if None, set to input dim
            act_layer=nn.ReLU,  # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

        # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = int(n_embd * expansion_factor)
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask

class LFE(nn.Module):
    def __init__(self, dim, cnn_in, kernel_size=3, MaxPool_ks=3, stride=1, k=2, num_heads=8, layer_len=2304, **kwargs,):
        super().__init__()
        self.stride = stride
        self.cnn_in = cnn_in
        self.pool_in = pool_in = dim - cnn_in
        
        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2
        
        self.proj1 = MaskedConv1D(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)        
        self.DWConv = MaskedConv1D(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False, groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()
        
        self.Maxpool = nn.MaxPool1d(kernel_size=MaxPool_ks, stride=stride, padding=MaxPool_ks//2)
        self.proj2 = MaskedConv1D(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

        self.qcontent_proj = nn.Linear(dim * 2, dim)
        
    def forward(self, x, mask):
        # B, C, L
        Q1 = x[:,:self.cnn_in,:].contiguous()
        Q1, mask = self.proj1(Q1, mask)
        Q1, mask = self.DWConv(Q1, mask)
        Q1 = self.mid_gelu1(Q1)
        
        Q2 = x[:,self.cnn_in:,:].contiguous()
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            mask = F.interpolate(
                mask.to(x.dtype), size=x.size(-1)//self.stride, mode='nearest')
        else:
            # masking out the features
            mask = mask
        Q2 = self.Maxpool(Q2)* mask.to(x.dtype)
        Q2, mask = self.proj2(Q2, mask)
        Q2 = self.mid_gelu2(Q2)
        
        Q = torch.cat((Q1, Q2), dim=1)
        Q = self.qcontent_proj(Q.permute(0, 2, 1))

        return Q, mask
 
class ConvFormer_block(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the input features
            DWConv_ks=3,
            n_ds_stride=1,  # downsampling stride for the current layer
            n_hidden=2048,
            MaxPool_ks=3,
            group=1,  # group for cnn
            attn_drop=0.,
            n_out=None,  # output dimension, if None, set to input dim
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            downsample_type='max',
            init_conv_vars=1,  # init gaussian variance for the weight
            layer_len = 2304,
            num_heads=8,
            LFE_lambda=0.5,
            use_mask=False,
            two_short_cut=False,
    ):
        super().__init__()
        self.stride = n_ds_stride
        self.layer_len = layer_len
        self.use_mask = use_mask
        self.two_short_cut = two_short_cut
        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1
        
        cnn_in = int(LFE_lambda * n_embd)
        self.LFE = LFE(n_embd, cnn_in, kernel_size=DWConv_ks, MaxPool_ks=MaxPool_ks, layer_len=layer_len)
        
        pos_embd = get_sinusoid_encoding(layer_len, n_embd) / (n_embd ** 0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)
        
        self.CAL = nn.MultiheadAttention(n_embd, num_heads, dropout=0.1)
        
        # two layer mlp
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        # self.reset_params(init_conv_vars=init_conv_vars)
    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.block.conv1.conv.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.block.proj1.conv.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.block.proj2.conv.weight, 0, init_conv_vars)
        # torch.nn.init.constant_(self.proj1.conv.bias, 0)
    
    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()
        B, C, T = x.shape
        if self.training:
            assert T <= self.layer_len, "Reached max length."
            pe = self.pos_embd
        if (not self.training):
            if T >= self.layer_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd 

        pe = (pe[:, :, :T] * out_mask.to(x.dtype))

        # ln_x = K = V = self.ln(x)
        ln_x = self.ln(x)

        Q, Q_mask = self.LFE(ln_x, out_mask)
        
        Z, _ = self.CAL(Q.permute(1, 0, 2), ln_x.permute(2, 0, 1), ln_x.permute(2, 0, 1))
        if self.two_short_cut:
            Z = Z + Q.permute(1, 0, 2)
        if self.training and (not self.use_mask):
            Z = Z.permute(1, 2, 0)
        else:
            Z = Z.permute(1, 2, 0) * Q_mask.to(x.dtype)
            
        out = x * out_mask + self.drop_path_out(Z)
        
        # FFN
        if self.training and (not self.use_mask):
            out = out + self.drop_path_mlp(self.mlp(self.gn(out)))
        else:
            out = out + self.drop_path_mlp(self.mlp(self.gn(out))) * out_mask

        return out, out_mask.bool()    



# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)