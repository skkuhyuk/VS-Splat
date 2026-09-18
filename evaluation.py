import os

n_thread = 1
os.environ["MKL_NUM_THREADS"] = f"{n_thread}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}" 
os.environ["OMP_NUM_THREADS"] = f"4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_thread}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}" 


from omegaconf import OmegaConf
import os, torch, math, imageio, cv2
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import sys,json
from lightning.system import system
from torch.utils.data import DataLoader
import pytorch_lightning as L
from dataLoader import dataset_dict

from pytorch_msssim import ssim
from tools.gen_video_path import uni_video_path,uni_mesh_path

import lpips
import torch.nn.functional as F
from tools.depth import acc_threshold,abs_error
from pytorch_lightning import seed_everything
def pos_to_index(position, grid_size=48, range_min=-0.5):
    position=torch.floor((position-range_min)*grid_size)
    return position
@torch.no_grad()
def main(cfg):
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    # data loader
    dataset = dataset_dict[cfg.infer.dataset.dataset_name]
    loader = DataLoader(dataset(cfg.infer.dataset), 
                              batch_size=cfg.infer.dataset.batch_size,
                              num_workers=cfg.infer.dataset.num_workers, 
                              shuffle=False,
                              pin_memory=False)
    loader_iter = iter(loader)

    device = 'cuda'
    my_system = system.load_from_checkpoint(cfg.infer.ckpt_path, cfg=cfg, map_location=device)

    # metrics
    lpips_vgg_fun = lpips.LPIPS(net='vgg').to(device)
    lpips_alex_fun = lpips.LPIPS(net='alex').to(device)

    names, depth_accs = [], []
    psnrs,ssims, lpips_vggs, lpips_alexs, occ_rates = [],[],[],[],[]
    os.makedirs(cfg.infer.save_folder, exist_ok=True)
    for i in tqdm(range(len(loader))):#len(loader)
        sample = next(loader_iter)
        sample = {key: tensor.to(device) if torch.is_tensor(tensor) else tensor for key, tensor in sample.items()} 

        my_system.net.eval()
        
        return_buffer = cfg.infer.video_frames > 0 or cfg.infer.save_mesh
        rendering_coarse, rendering_fine, occ_rate, save_center = my_system.net(sample, step=999999)
        name = sample['meta']['scene'][0].split('.')[0]
        images = rendering_fine["image"][0]
        img_gt = sample['tar_rgb'][0].permute(1,0,2,3).reshape(images.shape)
        save_centers = save_center[0]
        n_view = cfg.n_views

        
        if i<100:
            base_folder = os.path.join(cfg.infer.save_folder, name)
            gt_folder = os.path.join(base_folder, 'GT')
            pred_folder = os.path.join(base_folder, 'predict')
            os.makedirs(gt_folder, exist_ok=True)
            os.makedirs(pred_folder, exist_ok=True)
            ###################################################
            save_centers=pos_to_index(save_centers)
            centers_list = save_centers.detach().cpu().tolist()
            save_path=os.path.join(base_folder ,f"voxels.json")
            with open(save_path, 'w') as f:
                json.dump(centers_list, f, indent=2)
            #################################################
            img_gt_save=img_gt
            images_save=images
            img_gt_save=img_gt_save.detach().cpu().numpy()[...,::-1]*255
            images_save=images_save.detach().cpu().numpy()[...,::-1]*255
            for view_idx in range(8):
                gt_crop=img_gt_save[:, view_idx * 512:(view_idx + 1) * 512, :]
                img_crop=images_save[:, view_idx * 512:(view_idx + 1) * 512, :]
                if view_idx<4:
                    cv2.imwrite(os.path.join(gt_folder, f"context_{view_idx+1}.jpg"),gt_crop)
                    cv2.imwrite(os.path.join(pred_folder, f"context_{view_idx+1}.jpg"),img_crop)
                else:
                    cv2.imwrite(os.path.join(gt_folder, f"novel_{view_idx-3}.jpg"),gt_crop)
                    cv2.imwrite(os.path.join(pred_folder, f"novel_{view_idx-3}.jpg"),img_crop)          
            

        if cfg.infer.eval_novel_view_only:
            width = sample['meta']['tar_w']
            images = images.permute(2,0,1)[None][...,width*n_view:]
            img_gt = img_gt.permute(2,0,1)[None][...,width*n_view:]
        else:
            images = images.permute(2,0,1)[None]
            img_gt = img_gt.permute(2,0,1)[None]
        
        if images.shape[-1] > 0:
            color_loss_all = (images-img_gt)**2
            psnr = -10. * torch.log(color_loss_all.mean()) / torch.log(torch.tensor([10.]).to(device))
        
            ssim_val = ssim(images, img_gt, data_range=1.0, size_average=False)
            
            lpips_vgg = lpips_vgg_fun(img_gt*2-1,images*2-1)
            lpips_alex = lpips_alex_fun(img_gt*2-1,images*2-1)
            
            psnrs.append(psnr.item())
            ssims.append(ssim_val.item())
            lpips_vggs.append(lpips_vgg.item())
            lpips_alexs.append(lpips_alex.item())
            occ_rates.append(occ_rate.detach().cpu())

        names.append(name)
                
        del sample
    occ_rates=torch.stack(occ_rates, dim=0)
    if len(psnrs) and cfg.infer.metric_path is not None:
        print(f'evaluation score, psnr: {np.mean(psnrs)} ssim: {np.mean(ssims)}, lpips_vgg:{np.mean(lpips_vggs)}, lpips_alex: {np.mean(lpips_alexs)}, occ_rates:{occ_rates.mean(dim=0)}')
        scores = {'name':names, 'psnr':psnrs, 'ssim':ssims, \
                'lpips_vgg':lpips_vggs,'lpips_alex':lpips_alexs, \
                'depth_acc': depth_accs}
        scores.update({'psnr_mean':np.mean(psnrs), 'ssim_mean':np.mean(ssims), 
                    'lpips_vgg_mean':np.mean(lpips_vggs),'lpips_alex_mean':np.mean(lpips_alexs)})
        
        os.makedirs(os.path.dirname(cfg.infer.metric_path), exist_ok=True)
        with open(cfg.infer.metric_path, 'w') as f:
            json.dump(scores, f, indent=4)
        
if __name__ == '__main__':
    base_conf = OmegaConf.load('configs/vs-splat.yaml')
    path_config = sys.argv[1]
    cli_conf = OmegaConf.from_cli()
    second_conf = OmegaConf.load(path_config)
    cfg = OmegaConf.merge(base_conf, second_conf, cli_conf)

    main(cfg)