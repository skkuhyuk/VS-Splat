import math
import timm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor

from .utils import Normalize, AdaptiveNormalize, nonlinearity




def normalization(channels):
    return GroupNorm32(32, channels)
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class SparseGroupNorm(nn.Module):
    def __init__(self, num_groups, channels, eps=1e-6, affine=True):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, channels, eps=eps, affine=affine)

    def forward(self, input):
        feats = input.features
        coords = input.indices

        # B = coords[:,0].max().item() + 1
        B = input.batch_size
        C = feats.size(1)
        output_feats = torch.zeros_like(feats)
        
        for b in range(B):
            mask = (coords[:,0] == b)
            if mask.sum() == 0:
                continue

            b_feats = feats[mask] # N_b, C
            b_feats = b_feats.permute(1,0).unsqueeze(0) # 1, C, N_b
            b_feats_norm = self.group_norm(b_feats)
            b_feats_norm = b_feats_norm.squeeze(0).permute(1,0) # N_b, C
            output_feats[mask] = b_feats_norm
        return input.replace_feature(output_feats.contiguous())


class DinoWrapper(nn.Module):
    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly size
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model.forward_features(self.processor(image))
        if self.model_name == "vit_base_patch16_dinov3.lvd1689m":
            return outputs[:, 5:]
        else:
            return outputs[:, 1:]

        return outputs[:, 1:]
    
    def freeze(self, is_train: bool = False):
        print(f"\n======== image encoder is_train: {is_train} ========")
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
            data_config = timm.data.resolve_model_data_config(model)
            processor = transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err


class ModLN(nn.Module):
    def __init__(self, inner_dim, mod_dim, eps):
        super().__init__()
        
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(mod_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim * 2),
        )
    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1) # [N, D]
        return self.modulate(self.norm(x), shift, scale) # [N, L, D]

class TemperatureScheduler:
    def __init__(self, T0=1.0, Tmin=0.1, total_steps=20000, mode="exp"):
        self.T0 = T0
        self.Tmin = Tmin
        self.total_steps = total_steps
        self.mode = mode

        if mode == "exp":
            self.k = -math.log(self.Tmin / self.T0) / total_steps
        elif mode == "linear":
            self.alpha = (self.T0 - self.Tmin) / total_steps
        
    def get_temperature(self, step):
        if self.mode == "exp":
            T = self.T0 * math.exp(-self.k * step)
        elif self.mode == "linear":
            T = self.T0 - self.alpha * step
        else:
            raise ValueError
        
        return torch.tensor(max(T, self.Tmin))

class Embedder(nn.Module):
    """
    NeRF positional encoding
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()
    
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
        
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class PositionalEmbeddingNeRF(nn.Module):
    def __init__(
        self, multires=10,
    ):
        super().__init__()
        
        embed_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos]
        }
        self.pos_enc = Embedder(**embed_kwargs)

        # pos_dim = 6 * multires + 3
    
    def forward(self, x):
        return self.pos_enc(x)


class ResnetBlock3D(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h
    

def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    B, C, D, H, W = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.reshape(B, C_, D*scale_factor, H*scale_factor, W*scale_factor)
    return x

class Upsample3D(nn.Module):
    
    def __init__(self, in_channels, out_channels, mode='conv'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mode = mode
        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)
        elif mode == "nearest" or mode == "trilinear":
            assert in_channels == out_channels, 'Nearest mode requires in_channels to be equal to out_channels'
        
    def forward(self, x):
        if self.mode == "conv":
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        elif self.mode == "nearest":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        elif self.mode == "trilinear":
            return F.interpolate(x, scale_factor=2, mode="trilinear")

class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, mode="conv"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert in_channels == out_channels, "in_channels != out_channels"
    
    def forward(self, x):
        if self.mode == "conv":
            return self.conv(x)
        elif self.mode == "avgpool":
            return F.avg_pool3d(x, 2)


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
    
    def forward(self, input):
        return input.replace(super().forward(input.features))
    
class SubMConv3dResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, 
        dropout=0.0, indice_key=None
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = SparseGroupNorm(num_groups=32, channels=in_channels)
        # self.norm1 = LayerNorm32(in_channels, eps=1e-6)
        self.conv1 = spconv.SubMConv3d(in_channels, out_channels, 
                                       kernel_size=3, stride=1, padding=1, 
                                       indice_key=indice_key)
        self.norm2 = SparseGroupNorm(num_groups=32, channels=out_channels)
        # self.norm2 = LayerNorm32(out_channels, eps=1e-6)
        self.conv2 = spconv.SubMConv3d(out_channels, out_channels,
                                       kernel_size=3, stride=1, padding=1,
                                       indice_key=indice_key)
        
        self.skip_connection = SparseLinear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        # h = h.replace_feature(self.norm1(h.features))
        h = h.replace_feature(nonlinearity(h.features))
        h = self.conv1(h)
        h = self.norm2(h)
        # h = h.replace_feature(self.norm2(h.features))
        h = h.replace_feature(nonlinearity(h.features))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    


#################### Networks ####################
class VolumeNetwork(nn.Module):
    def __init__(
        self, in_channels, ch, ch_mult=(1,2), num_res_blocks=2, norm_type="group",
        out_channels=None, comp_in=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = ch if out_channels is None else out_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.comp_in = comp_in

        # conv in
        self.conv_in = nn.ModuleList()
        if self.comp_in:
            self.conv_in.append(nn.Conv3d(in_channels, 4*ch, kernel_size=1, stride=1, padding=0))
            self.conv_in.append(Normalize(4*ch, norm_type))
            self.conv_in.append(nn.SiLU())
            self.conv_in.append(nn.Conv3d(4*ch, ch, kernel_size=3, stride=1, padding=1))
        else:
            self.conv_in.append(nn.Conv3d(in_channels, ch, kernel_size=3, stride=1, padding=1))
        
        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        input_block_chans = [ch]
        self.down_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res
            res_block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock3D(block_in, block_out, norm_type=norm_type))
                input_block_chans.append(block_out)
                block_in = block_out
            conv_block.res = res_block
            # downsample
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample3D(block_in, block_in, "conv")
                input_block_chans.append(block_in)
            self.down_blocks.append(conv_block)
        
        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock3D(block_in, block_in, norm_type=norm_type))

        # upsampling
        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res
            res_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks+1):
                ich = input_block_chans.pop()
                res_block.append(ResnetBlock3D(block_in+ich, block_out, norm_type=norm_type))
                block_in = block_out
            conv_block.res = res_block
            # upsampling
            if i_level != 0:
                conv_block.upsample = Upsample3D(block_in, block_in, mode="trilinear")
            self.up_blocks.append(conv_block)
        
        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        hs = []

        # conv in
        h = x
        for block in self.conv_in:
            h = block(h)
        hs.append(h)

        # downsampling
        for i_level, block in enumerate(self.down_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
                hs.append(h)
            
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.up_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block.res[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class VoxelNetwork(nn.Module):
    def __init__(
        self, in_channels, ch, ch_mult=(1,2,3), num_res_blocks=2,
        out_channels=512, dino_dim=768, comp=False
    ):
        super().__init__()

        out_channels = ch if out_channels is None else out_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.comp = comp
        if self.comp:
            conv_in_dim = block_in
            self.comp_linear = nn.Linear(in_channels, block_in)
        else:
            conv_in_dim = in_channels
        
        self.conv_in = spconv.SubMConv3d(conv_in_dim, block_in, kernel_size=3, stride=1, padding=1)

        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # residual
            res_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(SubMConv3dResBlock(block_in, block_out))
                block_in = block_out
            conv_block.res = res_block
            self.conv_blocks.append(conv_block)
        
        # end
        self.norm_out = SparseGroupNorm(32, block_in + dino_dim)
        self.conv_out = spconv.SubMConv3d(block_in + dino_dim, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, feat_vol, proj_feat, occ, pos_encoding=None):
        B,C,D,H,W = feat_vol.shape

        batch_idx, z, y, x = torch.nonzero(occ, as_tuple=True)

        if pos_encoding != None:
            feats = torch.cat([feat_vol, pos_encoding], dim=1)
        else:
            feats = feat_vol
        feats = feats[batch_idx, :, z, y, x]
        if proj_feat != None:
            feats = torch.cat([proj_feat, feats], dim=1)

        if self.comp:
            feats = self.comp_linear(feats)

        indices = torch.stack([batch_idx, z, y, x], dim=1).int()
        spatial_shape = [D,H,W]

        with torch.cuda.amp.autocast(enabled=False):
            input_tensor = SparseConvTensor(
                features=feats.to(dtype=torch.float32),
                indices=indices,
                spatial_shape=spatial_shape,
                batch_size=B
            )

            if occ.sum() == 0:
                return input_tensor

            h = self.conv_in(input_tensor)

            for i_level, block in enumerate(self.conv_blocks):
                for i_block in range(self.num_res_blocks):
                    h = block.res[i_block](h)
            if proj_feat !=None:
                h = h.replace_feature(torch.cat([h.features, proj_feat], dim=1))
            h = self.norm_out(h)
            h = h.replace_feature(nonlinearity(h.features))
            h = self.conv_out(h)
        return h











