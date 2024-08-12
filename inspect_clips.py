import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scripts.etri_dataloaders import ETRIDataset
from torchvision import transforms as T


import matplotlib.pyplot as plt

from scripts.etri_dataloaders import ETRIDataset
from scripts.custom_transformations import repeat_color_channel, min_max_normalization, ConvertToUint8, ConvertToFloat32
from scripts.custom_transformations import ConvertToFloat64, sample_frames, PermuteDimensions



class ClipVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_clips = len(dataset)
        self.current_clip_idx = 0
        self.clip = dataset[self.current_clip_idx]

        self.num_frames = self.clip[0].shape[1]
        self.current_frame = 0

        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.clip[0][0][self.current_frame])
        self.ax.set_title(f'Clip {self.current_clip_idx + 1}/{self.num_clips}, Frame {self.current_frame + 1}/{self.num_frames}')

        axprev_frame = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bprev_frame = Button(axprev_frame, 'Prev Frame')
        self.bprev_frame.on_clicked(self.prev_frame)

        axnext_frame = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext_frame = Button(axnext_frame, 'Next Frame')
        self.bnext_frame.on_clicked(self.next_frame)

        axprev_clip = plt.axes([0.7, 0.15, 0.1, 0.075])
        self.bprev_clip = Button(axprev_clip, 'Prev Clip')
        self.bprev_clip.on_clicked(self.prev_clip)

        axnext_clip = plt.axes([0.81, 0.15, 0.1, 0.075])
        self.bnext_clip = Button(axnext_clip, 'Next Clip')
        self.bnext_clip.on_clicked(self.next_clip)

        plt.show()

    def update_frame(self):
        self.img.set_data(self.clip[0][0][self.current_frame])
        self.ax.set_title(f'Clip {self.current_clip_idx + 1}/{self.num_clips}, Frame {self.current_frame + 1}/{self.num_frames}')
        self.fig.canvas.draw()

    def prev_frame(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()

    def next_frame(self, event):
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.update_frame()

    def prev_clip(self, event):
        if self.current_clip_idx > 0:
            self.current_clip_idx -= 1
            self.clip = self.dataset[self.current_clip_idx]
            self.num_frames = self.clip[0].shape[1]
            self.current_frame = 0
            self.update_frame()

    def next_clip(self, event):
        if self.current_clip_idx < self.num_clips - 1:
            self.current_clip_idx += 1
            self.clip = self.dataset[self.current_clip_idx]
            self.num_frames = self.clip[0].shape[1]
            self.current_frame = 0
            self.update_frame()


    
#compose_transformations 
train_transform = T.Compose([
    min_max_normalization(scale_up=True),
    ConvertToFloat32(),
    PermuteDimensions(order=[3, 0, 1, 2]),
    repeat_color_channel(),
    sample_frames(nth=4),
])



train_set =  ETRIDataset(
    root_dir=r"/data/fhuemer/etri/masked_depthmaps",
    mode = "train",
    remove_background=False,
    transform=train_transform,
    single_camera=True,
    elders_only=False,
    max_number_frames = 200,
)


visualizer = ClipVisualizer(train_set)
