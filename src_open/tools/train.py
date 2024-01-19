
from pytorch_lightning.callbacks import ProgressBar
import pytorch_lightning as pl

import os
import torch
from ..dataset import get_dataset
from ..utils.lightening_utils import MyLightningLogger
from ..trainer.trainer import Trainer

def train(cfg):

    logger = MyLightningLogger('DeepAC', cfg.save_dir)
    logger.dump_cfg(cfg, 'train_cfg.yml')

    logger.info("Setting up data...")
    dataset = get_dataset(cfg.data.name)(cfg.data)
    train_data_loader = dataset.get_data_loader('train')
    if 'eval' not in cfg:
        val_data_loader = dataset.get_data_loader('val')
    else:
        val_dataset = get_dataset(cfg.eval.data.name)(cfg.eval.data)
        val_data_loader = val_dataset.get_data_loader('val')

    logger.info("Creating model...")
    task = Trainer(cfg, train_data_loader)
    
    # TODO: Load model
    if "load_model" in cfg:
        ckpt = torch.load(cfg.load_model)
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.models.load_model))

    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg
        else None
    )

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.trainer.total_epochs,
        gpus=cfg.device.gpu_ids,
        devices=len(cfg.device.gpu_ids),
        check_val_every_n_epoch=cfg.trainer.val_intervals,
        accelerator="gpu",  # "ddp",
        strategy="ddp",
        log_every_n_steps=cfg.trainer.log.interval,
        num_sanity_val_steps=0,
        resume_from_checkpoint=model_resume_path,
        callbacks=[ProgressBar(refresh_rate=0)],  # disable tqdm bar
        # plugins=DDPPlugin(find_unused_parameters=False),
        logger=logger,
        benchmark=True,
        # deterministic=True,
    )

    trainer.fit(model=task, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)