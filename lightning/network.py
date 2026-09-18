"""
Our code is based on 
https://github.com/autonomousvision/LaRa
"""

import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.quasirandom import SobolEngine
import pytorch_lightning as L

from tools.rsh import rsh_cart_3
from lightning.utils import MiniCam
from lightning.renderer import Renderer

from lightning.modules.utils import projection
from lightning.modules.modules import (DinoWrapper, ModLN, PositionalEmbeddingNeRF, TemperatureScheduler,
                                       VolumeNetwork, VoxelNetwork)



class ModelArgs:

    dino_name = "vit_base_patch16_224.dino" # ["vit_base_patch8_224.dino", "vit_base_patch16_224.dino", "vit_small_patch8_224.dino", "vit_small_patch16_224.dino", "vit_base_patch16_dinov3.lvd1689m"]
    mod_all = True

    volume_ch = 128
    volume_ch_mult = [1,2]

    voxel_ch = 256
    voxel_ch_mult = [1,1]
    use_pos_enc = True

    coarse_dim = 128
    fine_dim = 512

    mode = "gumbel"
    annealing_steps = 30000

    threshold = 0.5



class GaussianVolumeVoxelPredictor(L.LightningModule):
    def __init__(self, cfg, config=ModelArgs, white_bkgd=True):
        super().__init__()

        self.cfg = cfg
        self.scene_size = 0.5
        self.white_bkgd = white_bkgd
        self.n_predict = cfg.model.n_predict
        self.resolution = cfg.model.resolution

        self.use_pos_enc = config.use_pos_enc
        self.with_coarse = cfg.model.coarse
        self.with_fine = cfg.model.fine

        # build volume position
        self.feat_res = cfg.model.feature_resolution
        self.volume_res = cfg.model.volume_resolution
        self.register_buffer("centers_feat", self.build_dense_grid(self.feat_res))
        self.centers_feat = self.centers_feat.reshape(1,-1,3)
        self.register_buffer("centers_volume", self.build_dense_grid(self.volume_res))
        self.centers_volume = self.centers_volume.reshape(1,-1,3)

        # Feature extractor
        self.encoder = DinoWrapper(
            model_name=config.dino_name,
            is_train=True
        )
        if config.dino_name == "vit_small_patch16_224.dino" or config.dino_name == "vit_small_patch8_224.dino":
            encoder_feat_dim = 384
        elif config.dino_name == "vit_base_patch16_224.dino" or config.dino_name == "vit_base_patch8_224.dino":
            encoder_feat_dim = 768
        
        # Camera modulation
        self.mod_all = config.mod_all
        self.dir_norm = ModLN(encoder_feat_dim, mod_dim=16*2, eps=1e-6)

        # Positional encoding for skeletal primitive position
        if self.use_pos_enc:
            self.pos_enc = PositionalEmbeddingNeRF(multires=10)
            pos_channels = 63
        else: pos_channels = 0
        
        # 3D convolutional compressor & 3D U-Net
        self.volume_network = VolumeNetwork(in_channels=2*encoder_feat_dim, ch=config.volume_ch, ch_mult=config.volume_ch_mult,
                                            num_res_blocks=1, norm_type="group", out_channels=config.volume_ch, comp_in=True)
        # Selective Feature Refinement (SFR) module
        self.voxel_network = VoxelNetwork(in_channels=encoder_feat_dim+config.volume_ch+pos_channels, ch=config.voxel_ch, ch_mult=config.voxel_ch_mult,
                                          num_res_blocks=2, out_channels=config.fine_dim, comp=True)

        # Gumbel-Sigmoid temperature
        self.temp_scheduler = TemperatureScheduler(T0=1.0, Tmin=0.1, total_steps=config.annealing_steps, mode="linear")
        self.register_buffer("temperature", torch.tensor(1.0))

        # build predictors
        self.opacity_dim = 1
        self.rotation_dim = 4
        self.sh_dim = (cfg.model.sh_degree+1)**2*3
        self.scaling_dim = 2 if cfg.model.render == "2dgs" else 3

        self.coarse_dim = config.coarse_dim
        self.fine_dim = config.fine_dim

        self.threshold = config.threshold
        self.min_opacity = 1 / 255 / self.threshold

        self.predictor = Predictors(coarse_dim=self.coarse_dim, fine_dim=self.fine_dim,
                                    sh_dim=self.sh_dim, scaling_dim=self.scaling_dim, rotation_dim=self.rotation_dim, opacity_dim=self.opacity_dim,
                                    n_predict=self.n_predict, mode=config.mode)
        
        self.coarse_render = Renderer(sh_degree=cfg.model.sh_degree, white_background=white_bkgd, radius=1, render=cfg.model.render)
        self.fine_render = Renderer(sh_degree=cfg.model.sh_degree, white_background=white_bkgd, radius=1, render=cfg.model.render)

        # parameters initialization
        self.opacity_shift = -2.1792
        self.voxel_size = 1.0 / self.volume_res
        self.scaling_shift = np.log(np.exp(0.5*self.voxel_size/3.0)-1)
        div_offset = self.build_offset_Sobol(16)
        self.register_buffer("div_offset", div_offset)

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print(f"--- Feature Extractor Parameters: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad):,}")
        print(f"--- Coarse Volume Network Parameters: {sum(p.numel() for p in self.volume_network.parameters() if p.requires_grad):,}")
        print(f"--- Fine Voxel Network Parameters: {sum(p.numel() for p in self.voxel_network.parameters() if p.requires_grad):,}")
        print(f"--- Predictor Parameters: {sum(p.numel() for p in self.predictor.parameters() if p.requires_grad):,}\n")

        print(f"N Predict: {self.n_predict}\n")
    
    def build_dense_grid(self, reso):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(torch.meshgrid(array, array, array, indexing="ij"), dim=-1)
        grid = (grid + 0.5) / reso * 2 - 1
        return grid.reshape(reso, reso, reso, 3) * self.scene_size
    
    def build_offset_Sobol(self, N):
        """
        return (N,3)
        """
        sobol_engine = SobolEngine(dimension=3, scramble=True)
        points = sobol_engine.draw(N) * 2 - 1
        points = points.to(dtype=self.centers_volume.dtype) * self.voxel_size / 2
        return points * self.scene_size

    def build_feat_vol(self, src_inps, img_feats, n_views_sel, batch):

        h,w = src_inps.shape[-2:]

        src_w2cs = batch['tar_w2c'][:,:n_views_sel].reshape(-1,4,4)
        src_ixts = batch['tar_ixt'][:,:n_views_sel].reshape(-1,3,3)

        img_wh = torch.tensor([w,h], device=self.device)
        
        point_img, _ = projection(self.centers_feat, src_w2cs, src_ixts)
        point_img = (point_img + 0.5) / img_wh * 2 - 1.0

        if self.mod_all == False:
            # viewing direction
            rays = batch['tar_rays_down'][:,:n_views_sel]
            feats_dir = self.ray_to_plucker(rays).reshape(-1, *rays.shape[2:])
            feats_dir = torch.cat((rsh_cart_3(feats_dir[...,:3]),rsh_cart_3(feats_dir[...,3:6])),dim=-1)

            # features
            img_feats = torch.einsum('bchw->bhwc', img_feats)
            img_feats = self.dir_norm(img_feats, feats_dir)
            img_feats = torch.einsum('bhwc->bchw', img_feats)

        n_channels = img_feats.shape[1]
        feats_vol = F.grid_sample(img_feats.float(), point_img.unsqueeze(1), align_corners=False).to(img_feats)
        feats_vol = feats_vol.view(-1, n_views_sel, n_channels, self.feat_res, self.feat_res, self.feat_res)

        feats_vol_mean = feats_vol.mean(dim=1)
        feats_vol_var = feats_vol.var(dim=1, unbiased=False)
        feats_vol_mv = torch.cat([feats_vol_mean, feats_vol_var], dim=1)
        return feats_vol_mv


    def feature_projection(self, pos, feats, n_views_sel, batch, distance_weight=False):
        B,H,W = batch['tar_rgb'].shape[0], batch['tar_rgb'].shape[2], batch['tar_rgb'].shape[3]

        src_w2cs = batch['tar_w2c'][:,:n_views_sel]
        src_ixts = batch['tar_ixt'][:,:n_views_sel]

        img_wh = torch.tensor([W,H], device=self.device)

        feats_pos = []
        for b in range(B):
            point_img, point_z = projection(pos[b], src_w2cs[b], src_ixts[b])
            point_img = (point_img + 0.5) / img_wh * 2 - 1.0

            feats_pos_b = F.grid_sample(feats[b].float(), point_img.unsqueeze(-2), align_corners=False).to(feats)

            if distance_weight:
                point_z = point_z.squeeze(-1)
                inv_dist = 1.0 / (point_z + 1e-6)
                weights = inv_dist / inv_dist.sum(dim=0)
                feats_pos_b = weights.unsqueeze(1) * feats_pos_b.squeeze(-1)
                feats_pos_b = feats_pos_b.sum(dim=0).permute(1,0).contiguous()
            else:
                feats_pos_b = feats_pos_b.mean(dim=0).squeeze(-1).permute(1,0).contiguous()

            feats_pos.append(feats_pos_b)
        
        feats_pos = torch.cat(feats_pos, dim=0)
        return feats_pos
    
    def ray_to_plucker(self, rays):
        origin, direction = rays[...,:3], rays[...,3:6]
        # Normalize the direction vector to ensure it's a unit vector
        direction = F.normalize(direction, p=2.0, dim=-1)
        # Calculate the momentum vector (M = O x D)
        moment = torch.cross(origin, direction, dim=-1)
        # Plucker coordinates are L (direction) and M (moment)
        return torch.cat((direction, moment),dim=-1)
    
    def get_coarse_pt(self, offset, n_predict=1):
        B = offset.shape[0]
        cell_size = 0.5 * self.scene_size / self.volume_res
        coarse = self.centers_volume.unsqueeze(-2).expand(B,-1,n_predict,-1).reshape(offset.shape) + offset * cell_size
        return coarse

    def get_fine_pt_div(self, coarse, offset, n_predict):
        N = offset.shape[0]
        cell_size = 0.5 * self.scene_size / self.volume_res / 2
        fine = (coarse.unsqueeze(1).expand(-1,n_predict,-1) + self.div_offset.to(offset).clone().expand(N,-1,-1)).reshape(offset.shape) + offset * cell_size
        return fine

    def forward(self, batch, step):

        B,V,H,W,C = batch["tar_rgb"].shape
        if self.training:
            n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        else:
            n_views_sel = self.cfg.n_views

        _inps = batch["tar_rgb"][:,:n_views_sel].reshape(B*n_views_sel,H,W,C)
        _inps = torch.einsum("bhwc->bchw", _inps)

        # Feature extractor
        img_feats = torch.einsum("blc->bcl", self.encoder(_inps))
        token_size = int(np.sqrt(H*W/img_feats.shape[-1]))
        img_feats = img_feats.reshape(*img_feats.shape[:2], H//token_size, W//token_size)

        # Camera modulation
        if self.mod_all:
            rays = batch["tar_rays_down"][:,:n_views_sel]

            feats_dir = self.ray_to_plucker(rays).reshape(-1, *rays.shape[2:])
            feats_dir = torch.cat((rsh_cart_3(feats_dir[...,:3]),rsh_cart_3(feats_dir[...,3:6])),dim=-1)

            img_feats = torch.einsum("bchw->bhwc", img_feats)
            img_feats = self.dir_norm(img_feats, feats_dir)
            img_feats = torch.einsum("bhwc->bchw", img_feats)

        # Backprojection
        feat_vol = self.build_feat_vol(_inps, img_feats, n_views_sel, batch)

        # 3D convolutional compressor & 3D U-Net
        feat_vol = self.volume_network(feat_vol)
        feat_vol = torch.einsum("bcdhw->bdhwc", feat_vol)

        # Predict confidence logit
        act_mask = self.predictor.forward_act_mask(feat_vol)

        # Gumbel-Sigmoid
        self.temperature = self.temp_scheduler.get_temperature(step).to(act_mask)
        soft_act_mask, hard_act_mask = self.predictor.binary_thresholding(act_mask, temperature=self.temperature, threshold=self.threshold)

        act_rate = hard_act_mask.sum().float() / B / (self.volume_res ** 3) * 100

        # Predict coarse Gaussian primitives
        c_offset, c_shs, c_scaling, c_rotation, c_opacity = self.predictor.forward_coarse(feat_vol, self.opacity_shift, self.scaling_shift)
        c_centers = self.get_coarse_pt(c_offset, 1)
        c_opacity = self.coarse_render.opacity_activation(c_opacity).reshape(B,-1,1)
        soft_act_mask = soft_act_mask.reshape(B,-1,1)

        # Opacity refinement
        c_opacity = c_opacity * soft_act_mask

        # predict fine
        if self.with_fine:

            # Voxel selection
            hard_act_mask = hard_act_mask.squeeze(-1)
            hard_act_mask = hard_act_mask.reshape(B,-1)
            c_mask = hard_act_mask.bool()
            selected_c_centers = [c_centers[i][c_mask[i]] for i in range(c_centers.shape[0])]

            feat_vol = torch.einsum("bdhwc->bcdhw", feat_vol)
            img_feats = img_feats.reshape(B,n_views_sel,*img_feats.shape[1:])

            # Aggregated PAP-derived features
            projected_feats = self.feature_projection(selected_c_centers, img_feats, n_views_sel, batch, distance_weight=True)

            # Positional encoding for skeletal primitive position
            if self.use_pos_enc:
                pos_encoding = self.pos_enc(c_centers)
                pos_channel = pos_encoding.shape[-1]
                pos_encoding = pos_encoding.permute(0,2,1).reshape(B, pos_channel, *feat_vol.shape[-3:]).contiguous()
            else:
                pos_encoding = None
            
            hard_act_mask = hard_act_mask.reshape(B,self.volume_res,self.volume_res,self.volume_res)

            # Selective Feature Refinement (SFR) module
            fine_feats = self.voxel_network(feat_vol, projected_feats, hard_act_mask, pos_encoding=pos_encoding)
            fine_features = fine_feats.features

            # Fine primitive decoder
            f_offset, f_shs, f_scaling, f_rotation, f_opacity = self.predictor.forward_fine(fine_features, self.opacity_shift, self.scaling_shift)
            f_centers = self.get_fine_pt_div(c_centers[c_mask], f_offset, self.n_predict)
            f_opacity = self.fine_render.opacity_activation(f_opacity)

            # Skeletal primitive adjustor
            u_shs, u_opacity = self.predictor.forward_update(fine_features)
            u_opacity = torch.sigmoid(u_opacity) * c_opacity[c_mask]


        # Coarse & fine rendering
        rendering_coarse = []
        rendering_fine = []
        rendering_img_scale = batch.get("render_img_scale", 1.0)

        for i in range(B):
            znear, zfar = batch["near_far"][i]
            fovx, fovy = batch["fovx"][i], batch["fovy"][i]
            height, width = int(batch["meta"]["tar_h"][i] * rendering_img_scale), int(batch["meta"]["tar_w"][i] * rendering_img_scale)

            out_coarse = []
            out_fine = []
            tar_c2ws = batch["tar_c2w"][i]
            for j, c2w in enumerate(tar_c2ws):

                cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                rays_d = batch["tar_rays"][i,j]

                if self.with_coarse:
                    bg_color = batch['bg_color'][i,j]
                    self.coarse_render.set_bg_color(bg_color)

                    with torch.cuda.amp.autocast(enabled=False):
                        coarse = self.coarse_render.render_img(cam, rays_d, c_centers[i].float(), c_shs[i].float(), c_opacity[i].float(), c_scaling[i].float(), c_rotation[i].float(), self.device, prex="_coarse")
                    out_coarse.append(coarse)
                
                if self.with_fine:
                    bg_color = batch['bg_color'][i,j]
                    self.fine_render.set_bg_color(bg_color)

                    f_mask_i = (fine_feats.indices[:,0]==i)
                    f_centers_i = f_centers[f_mask_i].view(-1,3)
                    f_shs_i = f_shs[f_mask_i].view(-1,self.sh_dim//3,3)
                    f_opacity_i = f_opacity[f_mask_i].view(-1,self.opacity_dim)
                    f_scaling_i = f_scaling[f_mask_i].view(-1,self.scaling_dim)
                    f_rotation_i = f_rotation[f_mask_i].view(-1,self.rotation_dim)
 
                    hard_act_mask = hard_act_mask.reshape(B,-1)
                    c_mask_i = hard_act_mask[i].bool()
                    f_centers_i = torch.cat([f_centers_i, c_centers[i][c_mask_i]], dim=0)
                    f_shs_i = torch.cat([f_shs_i, u_shs[f_mask_i]], dim=0)
                    f_opacity_i = torch.cat([f_opacity_i, u_opacity[f_mask_i]], dim=0)
                    f_scaling_i = torch.cat([f_scaling_i, c_scaling[i][c_mask_i]], dim=0)
                    f_rotation_i = torch.cat([f_rotation_i, c_rotation[i][c_mask_i]], dim=0)

                    with torch.cuda.amp.autocast(enabled=False):
                        if c_mask_i.sum() == 0:
                            fine = self.fine_render.render_empty_img(rays=rays_d, device=self.device)
                        else:
                            fine = self.fine_render.render_img(cam, rays_d, f_centers_i.float(), f_shs_i.float(), f_opacity_i.float(), f_scaling_i.float(), f_rotation_i.float(), self.device)
                    out_fine.append(fine)
            
            if self.with_coarse:
                rendering_coarse.append({k: torch.cat([d[k] for d in out_coarse], dim=1) for k in out_coarse[0]})
            else: rendering_coarse = None

            if self.with_fine:
                rendering_fine.append({k: torch.cat([d[k] for d in out_fine], dim=1) for k in out_fine[0]})
            else: rendering_fine = None

        if rendering_coarse:
            rendering_coarse = {k: torch.stack([d[k] for d in rendering_coarse]) for k in rendering_coarse[0]}
        if rendering_fine:
            rendering_fine = {k: torch.stack([d[k] for d in rendering_fine]) for k in rendering_fine[0]}

        if self.cfg.infer:
            return rendering_coarse, rendering_fine, act_rate, selected_c_centers
        else:
            return rendering_coarse, rendering_fine, act_rate
    

# ==================== Predictor ====================
class Predictors(nn.Module):
    def __init__(
        self, coarse_dim, fine_dim,
        sh_dim, scaling_dim, rotation_dim, opacity_dim,
        n_predict=3, mode="gumbel",
    ):
        super().__init__()

        self.n_predict = n_predict
        
        self.coarse_dim = coarse_dim
        self.appear_dim = fine_dim

        self.mode = mode

        self.sh_dim = sh_dim
        self.opacity_dim = opacity_dim
        self.scaling_dim = scaling_dim
        self.rotation_dim = rotation_dim

        num_layer = 2

        act_layers = [nn.Linear(coarse_dim, coarse_dim), nn.SiLU()] + \
            [nn.Linear(coarse_dim, coarse_dim), nn.SiLU()] * (num_layer - 1) + \
            [nn.Linear(coarse_dim, 1)]
        self.act_predictor = nn.Sequential(*act_layers)

        self.coarse_out = 3 + sh_dim + opacity_dim + scaling_dim + rotation_dim
        coarse_layers = [nn.Linear(coarse_dim, coarse_dim), nn.SiLU()] + \
            [nn.Linear(coarse_dim, coarse_dim), nn.SiLU()] * (num_layer - 1) + \
            [nn.Linear(coarse_dim, self.coarse_out)]
        self.coarse_predictor = nn.Sequential(*coarse_layers)

        self.fine_out = 3 + sh_dim + opacity_dim + scaling_dim + rotation_dim
        fine_layers = [nn.Linear(fine_dim, fine_dim), nn.SiLU()] + \
            [nn.Linear(fine_dim, fine_dim), nn.SiLU()] * (num_layer - 1) + \
            [nn.Linear(fine_dim, self.n_predict * self.fine_out)]
        self.fine_predictor = nn.Sequential(*fine_layers)

        self.update_out = 1 + sh_dim # opacity, sh
        update_layers = [nn.Linear(fine_dim, fine_dim), nn.SiLU()] + \
            [nn.Linear(fine_dim, fine_dim), nn.SiLU()] * (num_layer - 1) + \
            [nn.Linear(fine_dim, self.update_out)]
        self.update_predictor = nn.Sequential(*update_layers)

        self.init(self.act_predictor)
        self.init(self.coarse_predictor)
        self.init(self.fine_predictor)
        self.init(self.update_predictor)

    
    def init(self, layers):
        # MLP initialization as in mipnerf360
        init_method = "xavier"
        if init_method:
            for layer in layers:
                if not isinstance(layer, torch.nn.Linear):
                    continue
                if init_method == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                elif init_method == "xavier":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)


    def binary_thresholding(self, logits, temperature=0.5, threshold=0.5, eps=1e-10):
        # Apply Gumbel-Sigmoid
        dtype = logits.dtype
        with torch.no_grad():
            # generate a random sample from the uniform distribution
            if self.mode == "gumbel":
                uniform1 = torch.rand(logits.size(), dtype=dtype)
                uniform2 = torch.rand(logits.size(), dtype=dtype)
                noise = -torch.log(torch.log(uniform1 + eps) / torch.log(uniform2 + eps) + eps).cuda()
            elif self.mode == "concrete":
                uniform = torch.rand(logits.size(), dtype=dtype)
                noise = (torch.log(uniform + eps) - torch.log(1 - uniform + eps)).cuda()
        reparam = (logits + noise) / temperature
        y_soft = torch.sigmoid(reparam)

        # straight-through estimator
        y_hard = (y_soft > threshold).to(dtype=dtype)
        return y_soft, (y_hard - y_soft).detach() + y_soft
    
    def forward_act_mask(self, feats):
        act_mask = self.act_predictor(feats)
        return act_mask
    
    def forward_coarse(self, feats, opacity_shift, scaling_shift):
        parameters = self.coarse_predictor(feats).float()
        parameters = parameters.view(*parameters.shape[:-1], 1, -1)
        offset, sh, opacity, scaling, rotation = torch.split(
            parameters,
            [3, self.sh_dim, self.opacity_dim, self.scaling_dim, self.rotation_dim],
            dim=-1
        )
        opacity = opacity + opacity_shift
        scaling = scaling + scaling_shift
        offset = torch.sigmoid(offset) * 2 - 1.0

        B = opacity.shape[0]
        opacity = opacity.view(B, -1, self.opacity_dim)
        scaling = scaling.view(B, -1, self.scaling_dim)
        rotation = rotation.view(B, -1, self.rotation_dim)
        offset = offset.view(B, -1, 3)
        sh = sh.view(B, -1, self.sh_dim//3, 3)
        return offset, sh, scaling, rotation, opacity

    def forward_fine(self, feats, opacity_shift, scaling_shift):
        parameters = self.fine_predictor(feats)
        parameters = parameters.view(*parameters.shape[:-1], self.n_predict, -1)
        offset, sh, opacity, scaling, rotation = torch.split(
            parameters,
            [3, self.sh_dim, self.opacity_dim, self.scaling_dim, self.rotation_dim],
            dim=-1
        )
        opacity = opacity + opacity_shift
        scaling = scaling + scaling_shift
        offset = torch.sigmoid(offset) * 2 - 1.0

        sh = sh.view(*sh.shape[:-1], self.sh_dim//3,3)

        return offset, sh, scaling, rotation, opacity

    def forward_update(self, feats):
        parameters = self.update_predictor(feats)
        parameters = parameters.view(*parameters.shape[:-1], 1, -1)
        sh, opacity = torch.split(
            parameters,
            [self.sh_dim, 1],
            dim=-1
        )

        sh = sh.view(-1, self.sh_dim//3, 3)
        opacity = opacity.view(-1, self.opacity_dim)
        return sh, opacity



