import os

gpu_id = 0
name = 'test'
ckpt_path = "" # path of "epoch=49.ckpt"
gobj_dataset_path="" #path of dataset(~/gobjaverse.h5)
gso_dataset_path="" #path of dataset(~/google_scanned_object)
co3d_teddybear_dataset_path="" #path of dataset(~/co3d_teddybear.h5)
co3d_hydrant_dataset_path="" #path of dataset(~/co3d_hydrant.h5)
for n_views in [4]:

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_GSO_{n_views}_views.json ' \
        f'infer.dataset.dataset_name=GSO infer.dataset.data_root={gso_dataset_path} infer.eval_depth=[0.005,0.01,0.02] ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs/image_vis/{name}_GSO_{n_views}_views infer.dataset.n_group={n_views} '
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_gobjeverse_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root={gobj_dataset_path} ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs/image_vis/{name}_gobjaverse_{n_views}_views infer.dataset.n_group={n_views} '
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_co3d_teddybear_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root={co3d_teddybear_dataset_path} ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs/image_vis/{name}_co3d_teddybear infer.dataset.n_group={n_views} '
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_co3d_hydrant_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root={co3d_hydrant_dataset_path} ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs/image_vis/{name}_co3d_hydrant infer.dataset.n_group={n_views} '
    os.system(cmd)
    
