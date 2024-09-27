print("Dont forget to set CUDA_VISIBLE_DEVICES")
import os

cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
import pickle
import inspect

from model import TubeViTLightningModule
import click
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
#from pytorchvideo.transforms import Normalize, Permute, RandAugment
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
import torch
from mrs_dataloaders import MRSActivityDataset
from custom_transformations import repeat_color_channel, min_max_normalization, ConvertToFloat32, sample_frames, PermuteDimensions,ConvertToUint8, ConvertToFloat64
from torchvision.models import ViT_B_16_Weights
from torch.nn import functional as F
from datetime import datetime


import yaml
import ast
import wandb

#from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint

@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")
@click.option("--config", type=click.Path(exists=True), help="path to config file.")
@click.option("-nc", "--num-classes", type=int, default=16, help="num of classes of dataset.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(640, 480), help="Height and width of image.")
@click.option("--fast-dev-run", type=bool, is_flag=True, show_default=True, default=False)
@click.option("--seed", type=int, default=42, help="random seed.")

def main(
    dataset_root,
    num_classes,
    video_size,
    fast_dev_run,
    seed,
    config,
):
    config_path = config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)


    torch.set_float32_matmul_precision(config["torch"]["precision"])

    transformation_mapping = {
        'min_max_normalization': min_max_normalization,
        'repeat_color_channel': repeat_color_channel,
        'ConvertToUint8': ConvertToUint8,
        'ConvertToFloat32': ConvertToFloat32,
        'ConvertToFloat64': ConvertToFloat64,
        'sample_frames': sample_frames,
        'PermuteDimensions': PermuteDimensions,
    }

    def compose_transformations(transformations):
        applied_transformations = []
        for transform in transformations:
            transform_type = transform.pop('type')
            if transform_type in transformation_mapping:
                transform_function = transformation_mapping[transform_type]
                applied_transformations.append(transform_function(**transform))
            else:
                raise ValueError(f"Transformation {transform_type} not found.")

        composed_transform = T.Compose(applied_transformations)
        return composed_transform
    
    train_transform = compose_transformations(config['transforms']['train'])
    test_transform = compose_transformations(config['transforms']['test'])


    train_set =  MRSActivityDataset(
        root_dir=dataset_root,
        mode = "train",
        remove_background=config["ETRI_dataset"]["remove_background"],
        transform=train_transform,
        max_number_frames = config["ETRI_dataset"]["max_number_frames"],
    )

    val_set =  MRSActivityDataset(
        root_dir=dataset_root,
        mode = "val",
        remove_background=config["ETRI_dataset"]["remove_background"],
        transform=test_transform,
        max_number_frames = config["ETRI_dataset"]["max_number_frames"],
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=config["data_loader"]["batch_size"],
        num_workers=config["data_loader"]["num_workers"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=config["data_loader"]["batch_size"],
        num_workers=config["data_loader"]["num_workers"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    x, _ = next(iter(train_dataloader))

    model = TubeViTLightningModule(
        num_classes=num_classes,
        video_shape=x.shape[1:],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        hidden_dim=config["model"]["hidden_dim"],
        mlp_dim=config["model"]["mlp_dim"],
        lr=config["model"]["learning_rate"],
        weight_decay=config["model"]["weight_decay"],
        weight_path=os.path.join("..","saved_weights","nc200_3color_channels.pt"),
        max_epochs=config["model"]["max_epochs"],
        kernel_sizes=config["model"]["kernel_sizes"],
        strides=config["model"]["strides"],
        offsets=config["model"]["offsets"],
        use_pretrained=config["model"]["use_pretrained"],
        use_pretrained_conv=config["model"]["use_pretrained_conv"],
        freeze_first_layers=config["model"]["freeze_first_layers"],
    )

    #get current date and hour
    current_date = datetime.now().strftime("%d-%m-%Y_%H")

    wandb_logger = WandbLogger(
        project="sparse_tubes_msr", 
        name=f"Tube_{config_path}_{current_date}",
        config=config,)

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=config["trainer"]["save_checkpoint_path"],  # Specify the directory
        )
    ]

    trainer = pl.Trainer(
        max_epochs=config["model"]["max_epochs"],
        accelerator="auto",
        fast_dev_run=fast_dev_run,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,ckpt_path=config["trainer"]["load_checkpoint_path"])


if __name__ == "__main__":
    main()     

