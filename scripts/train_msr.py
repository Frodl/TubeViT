print("Dont forget to set CUDA_VISIBLE_DEVICES")
from model import TubeViTLightningModule
import os
import pickle
import inspect

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
from custom_transformations import repeat_color_channel, min_max_normalization, ConvertToFloat32, sample_frames, PermuteDimensions
from torchvision.models import ViT_B_16_Weights
from torch.nn import functional as F


import ast
import wandb

#from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint

@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset.")


@click.option("-nc", "--num-classes", type=int, default=55, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=8, help="batch size.")
@click.option("-m", "--max_number_frames", type=int, default=150, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(512, 424), help="Height and width of image.")
@click.option("-s", "--presample", type=int, default=4, help="pre sample frames from video")
@click.option("--strides", help="sampeling for tubes")
@click.option("--max-epochs", type=int, default=10, help="max epochs.")
@click.option("--num_workers", type=int, default=0, help="num workers.")
@click.option("--fast-dev-run", type=bool, is_flag=True, show_default=True, default=False)
@click.option("--seed", type=int, default=42, help="random seed.")
@click.option("--preview-video", type=bool, is_flag=True, show_default=True, default=False, help="Show input video")
@click.option("--use_pretrained", type=bool, is_flag=True, show_default=True, default=False, help="Whether to use pretrained encoder")
@click.option("--use_pretrained_conv", type=bool, is_flag=True, show_default=True, default=False, help="Whether to use pretrained encoder")

def main(
    dataset_root,
    num_classes,
    batch_size,
    video_size,
    presample,
    strides,
    max_epochs,
    num_workers,
    fast_dev_run,
    seed,
    preview_video,
    max_number_frames,
    use_pretrained,
    use_pretrained_conv,
):
    pl.seed_everything(seed)

    if strides is not None:
        strides = ast.literal_eval(strides)

    train_transform = T.Compose(
        [   min_max_normalization(), # also transforms to float64
            ConvertToFloat32(), # reduce to float32 in order to save memory
            PermuteDimensions((3, 0, 1, 2)),  # from (T, H, W, C) to (C, T, H, W)
            repeat_color_channel(), 
            sample_frames(nth=presample)
        ]
    )
    

    test_transform = T.Compose(
        [   min_max_normalization(),
            ConvertToFloat32(), 
            PermuteDimensions((3, 0, 1, 2)),  # from (T, H, W, C) to (C, T, H, W)
            repeat_color_channel(), 
            sample_frames(nth=presample)
        ]
    )

    train_set =  MRSActivityDataset(
        root_dir=dataset_root,
        mode = "train",
        transform=train_transform,
        get_metadata=True,
        max_number_frames = max_number_frames,
    )



    val_set =  MRSActivityDataset(
        root_dir=dataset_root,
        mode = "val",
        transform=test_transform,
        max_number_frames = max_number_frames,
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    x, y = next(iter(train_dataloader))
    print(x.shape)

    hidden_dim=768#//4
    mlp_dim=3072#//4

    model = TubeViTLightningModule(
        num_classes=num_classes,
        video_shape=x.shape[1:],
        num_layers=12,
        num_heads=12,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        lr=1e-4,
        weight_decay=0.001,
        weight_path=os.path.join("..","saved_weights","nc200_3color_channels.pt"),
        max_epochs=max_epochs,
        strides=strides,
        use_pretrained=use_pretrained,
        use_pretrained_conv=use_pretrained_conv,
    )

    #get dictionary of arguments for name of wandb run
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    arguments_dict = {arg: values[arg] for arg in args}

    wandb_logger = WandbLogger(
        project="sparse_tubes_msr", 
        name=f"TubeViT_minmaxscaling_no_scaleup_{batch_size}_{max_number_frames}_{presample}_{max_epochs}_{num_workers}_{hidden_dim}_{mlp_dim}_{use_pretrained}_{use_pretrained_conv}",
        config=arguments_dict,)

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='./model_checkpoints/',  # Specify the directory
        )
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        fast_dev_run=fast_dev_run,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    #trainer.save_checkpoint("./models/tubevit_ucf101.ckpt")



def convert_conv_weight(model):
    weights = ViT_B_16_Weights.DEFAULT.get_state_dict(progress=True)

    # inflated vit path convolution layer weight
    conv_proj_weight = weights["conv_proj.weight"]
    conv_proj_weight = F.interpolate(conv_proj_weight, (8, 8), mode="bilinear")
    conv_proj_weight = torch.unsqueeze(conv_proj_weight, dim=2)
    conv_proj_weight = conv_proj_weight.repeat(1, 1, 8, 1, 1)
    conv_proj_weight = conv_proj_weight / 8.0

    # remove missmatch parameters
    weights.pop("encoder.pos_embedding")
    weights.pop("heads.head.weight")
    weights.pop("heads.head.bias")

    model.load_state_dict(weights, strict=False)
    model.sparse_tubes_tokenizer.conv_proj_weight = torch.nn.Parameter(conv_proj_weight, requires_grad=True)

    return model



if __name__ == "__main__":
    main()     

