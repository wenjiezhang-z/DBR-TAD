import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (get_sinusoid_encoding, MaskedConv1D, ConvBlock, LayerNorm, ConvFormer_block)
from .models import register_backbone
from functools import partial
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.layers.helpers import to_2tuple
   
@register_backbone("ConvFormer")
class ConvFormer(nn.Module):
    """
        A backbone consisting of LFE, CAL, and FFN.
    """

    def __init__(
            self,
            n_in,  # input feature dimension
            n_embd,  # embedding dimension (after convolution)
            block_mlp_dim,  # the numnber of dim in SGP
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            path_pdrop=0.0,  # droput rate for drop path
            downsample_type='max',  # 'max/avg' how to downsample feature in FPN
            DWConv_ks=[-1] * 6,
            MaxPool_ks=[3] * 6,
            init_conv_vars=1,  # initialization of gaussian variance for the weight in SGP
            use_abs_pe=False,  # use absolute position embedding
            LFE_lambda=0.5,
            use_mask=False,
            two_short_cut=False,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(DWConv_ks) == (1 + arch[2])
        self.arch = arch
        self.DWConv_ks = DWConv_ks
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())
        
        # stem network
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                ConvFormer_block(n_embd=n_embd, DWConv_ks=self.DWConv_ks[0], n_ds_stride=1, MaxPool_ks=MaxPool_ks[0], n_hidden=block_mlp_dim,
                        group=1, path_pdrop=path_pdrop, downsample_type=downsample_type, init_conv_vars=init_conv_vars,LFE_lambda=LFE_lambda,use_mask=use_mask,two_short_cut=two_short_cut))

        # main branch using ConvFormer block with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            layer_len = self.max_len // (2 ** (idx+1))
            
            self.branch.append(ConvFormer_block(n_embd=n_embd, DWConv_ks=self.DWConv_ks[1 + idx], n_ds_stride=self.scale_factor, n_hidden=block_mlp_dim, MaxPool_ks=MaxPool_ks[1 + idx],
                                    group=1, path_pdrop=path_pdrop, downsample_type=downsample_type, init_conv_vars=init_conv_vars, layer_len=layer_len,LFE_lambda=LFE_lambda,use_mask=use_mask,two_short_cut=two_short_cut))
        
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module, init_conv_vars=0.1):
        # set nn.Linear/nn.Conv1d bias term to 0
        # print(module)
        if isinstance(module, (nn.Linear)):
            # trunc_normal_(module.weight, std=.001)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        if isinstance(module, nn.Conv1d):
            # trunc_normal_(module.weight, std=.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))
            
        # training: using fixed length position embeddings, 
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem network
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)
        
        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x,)
        out_masks += (mask,)
        # main branch with downsampling
        for idx in range(len(self.branch)): 
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
    

@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """

    def __init__(
            self,
            n_in,  # input feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_embd_ks,  # conv kernel size of the embedding network
            arch=(2, 2, 5),  # (#convs, #stem convs, #branch convs)
            scale_factor=2,  # dowsampling rate for the branch
            with_ln=False,  # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x,)
        out_masks += (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
            
            

