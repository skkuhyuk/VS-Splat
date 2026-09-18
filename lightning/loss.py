import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

import numpy as np
import lpips as lpips_lib
from pytorch_msssim import MS_SSIM


class Losses(nn.Module):
    def __init__(self, cfg):
        super(Losses, self).__init__()

        self.cfg = cfg
        
        self.lambda_l2 = cfg.train.lambda_l2
        self.lambda_l1 = cfg.train.lambda_l1
        self.lambda_ssim = cfg.train.lambda_ssim
        self.lambda_lpips = cfg.train.lambda_lpips

        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        if self.lambda_lpips != 0:
            self.lpips_fn = lpips_lib.LPIPS(net='vgg').requires_grad_(False)

    def forward(self, batch, out_coarse, out_fine, act, iter):

        scalar_stats = {}
        loss = 0

        B,V,H,W = batch['tar_rgb'].shape[:-1]

        if out_coarse:
            if 'image_coarse' in out_coarse:
                
                tar_rgb = batch['tar_rgb'].permute(0,2,1,3,4).reshape(B,H,V*W,3)

                with autocast(enabled=False):
                    coarse_mse_loss = (out_coarse[f'image_coarse']-tar_rgb)**2

                    if self.lambda_l1 != 0:
                        coarse_mae_loss = torch.abs(out_coarse[f'image_coarse']-tar_rgb)
                        loss = loss + self.lambda_l1 * coarse_mae_loss.mean()
                        scalar_stats.update({f"coarse_mae": coarse_mae_loss.detach().mean()})
                    if self.lambda_l2 != 0:
                        loss = loss + self.lambda_l2 * coarse_mse_loss.mean()
                        scalar_stats.update({f"coarse_mse": coarse_mse_loss.detach().mean()})

                    coarse_psnr = -10. * torch.log(coarse_mse_loss.detach().mean()) / \
                        torch.log(torch.Tensor([10.]).to(coarse_mse_loss.device))
                    scalar_stats.update({f'coarse_psnr': coarse_psnr})

                    pred_coarse = out_coarse["image_coarse"].reshape(B,H,V,W,3).permute(0,2,4,1,3).reshape(B*V,3,H,W).contiguous().float()
                    tar_coarse = tar_rgb.reshape(B,H,V,W,3).permute(0,2,4,1,3).reshape(B*V,3,H,W).contiguous().float()
                    if self.lambda_ssim != 0:
                        coarse_ssim_val = self.ssim(pred_coarse, tar_coarse)
                        loss = loss + self.lambda_ssim * (1 - coarse_ssim_val)
                        scalar_stats.update({f"coarse_ssim": coarse_ssim_val.detach()})
        
        if out_fine:
            if 'image' in out_fine:

                tar_rgb = batch['tar_rgb'].permute(0,2,1,3,4).reshape(B,H,V*W,3)

                with autocast(enabled=False):
                    mse_loss = (out_fine[f'image']-tar_rgb)**2

                    if self.lambda_l2 != 0:
                        loss = loss + self.lambda_l2 * mse_loss.mean()
                        scalar_stats.update({f"mse": mse_loss.detach().mean()})

                    if self.lambda_l1 != 0:
                        mae_loss = torch.abs(out_fine[f'image']-tar_rgb)
                        loss = loss + self.lambda_l1 * mae_loss.mean()
                        scalar_stats.update({f"mae": mae_loss.detach().mean()})

                    psnr = -10. * torch.log(mse_loss.detach().mean()) / \
                        torch.log(torch.Tensor([10.]).to(mse_loss.device))
                    scalar_stats.update({f'psnr': psnr})
                    
                    pred_rgb = out_fine[f"image"].reshape(B,H,V,W,3).permute(0,2,4,1,3).reshape(B*V,3,H,W).contiguous().float() # b h vw 3 -> b h v w 3 -> b v 3 h w -> bv 3 h w
                    tar_rgb = tar_rgb.reshape(B,H,V,W,3).permute(0,2,4,1,3).reshape(B*V,3,H,W).contiguous().float() # b h vw 3 -> b h v w 3 -> b v 3 h w -> bv 3 h w
                    if self.lambda_lpips != 0:
                        lpips = torch.mean(
                            self.lpips_fn(pred_rgb * 2 - 1, 
                                          tar_rgb * 2 - 1)  
                        )
                        loss = loss + self.lambda_lpips * lpips
                        scalar_stats.update({f"lpips": lpips.detach()})
                    
                    if self.lambda_ssim != 0:
                        ssim_val = self.ssim(pred_rgb, tar_rgb)
                        loss = loss + self.lambda_ssim * (1 - ssim_val)
                        scalar_stats.update({f"ssim": ssim_val.detach()})

        scalar_stats.update({f'activation rate': act.detach()})
        scalar_stats.update({f'total_loss': loss})
 
        return loss, scalar_stats

 