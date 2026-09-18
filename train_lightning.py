import os

n_thread = 4
os.environ["MKL_NUM_THREADS"] = f"{n_thread}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}" 
os.environ["OMP_NUM_THREADS"] = f"4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_thread}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}" 


import torch
from dataLoader import dataset_dict
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
import pytorch_lightning as L
from einops import rearrange
from datetime import datetime

from lightning.system import system
from pytorch_lightning.profilers import PyTorchProfiler 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything



def main(cfg):
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    # data loader
    train_dataset = dataset_dict[cfg.train_dataset.dataset_name]
    train_loader = DataLoader(train_dataset(cfg.train_dataset), 
                              batch_size=cfg.train.batch_size,
                              num_workers=8, 
                              shuffle=True,
                              pin_memory=True,)
    val_dataset = dataset_dict[cfg.test_dataset.dataset_name]
    val_loader = DataLoader(val_dataset(cfg.test_dataset), 
                              batch_size=cfg.test.batch_size,
                              num_workers=8, 
                              shuffle=False,
                              pin_memory=False,)
    
    # build logger
    project_name = cfg.exp_name.split("/")[0]
    exp_name = cfg.exp_name.split("/")[1]

    os.makedirs(cfg.logger.dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if cfg.logger.name == "tensorboard":
        logger = TensorBoardLogger(save_dir=cfg.logger.dir, name=exp_name)
    elif cfg.logger.name == "wandb":
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        logger = WandbLogger(name=exp_name, project=project_name, save_dir=cfg.logger.dir)
    
    # Set up ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.logger.dir,        # Path where checkpoints will be saved
        filename='{epoch}',        # Filename for the checkpoints
        save_top_k=-1,             # Set to -1 to save all checkpoints
        every_n_epochs=1,          # Save a checkpoint every K epochs
        save_on_train_epoch_end=True,  # Ensure it saves at the end of an epoch, not the beginning
    )

    my_system = system(cfg)

    trainer = L.Trainer(devices=cfg.gpu_id,
                        num_nodes=1,
                        max_epochs=cfg.train.n_epoch,
                        accelerator='gpu',
                        strategy=DDPStrategy(find_unused_parameters=False),
                        accumulate_grad_batches=2,
                        logger=logger,
                        gradient_clip_val=0.5,
                        precision="bf16-mixed",
                        callbacks=[checkpoint_callback],
                        log_every_n_steps=10,
                        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
                        limit_val_batches=cfg.train.limit_val_batches,  # Run on only 10% of the validation data
                        limit_train_batches=cfg.train.limit_train_batches,
                        )
    

    t0 = datetime.now()
    trainer.fit(
        my_system, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.model.ckpt_path
        )
    
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))


if __name__ == '__main__':
    
    base_conf = OmegaConf.load('configs/vs-splat.yaml')

    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)
    
    main(cfg)