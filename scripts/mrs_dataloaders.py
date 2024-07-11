### Dataloaders for ETRI dataset ###
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import math

# train = 2/3
# val = 1/6
# test = 1/6

class MRSActivityDataset(Dataset):

    def __init__(self, root_dir,max_number_frames, mode = "train" ,transform=None, eval_mode = "cs", get_metadata = False) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.labels =[]
        self.clips = []
        self.room = []
        self.edition = []
        self.subjects = []
        self.max_number_frames = max_number_frames
        self.get_metadata = get_metadata
        self.meta_data = {
            "clip_lengths": [],
            "filenames": [],
            "labels": [],}

        self.subject_selector()

        #walk through root_dir and create list of images and labels
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                naming_list = file.split('_')

                action = int(naming_list[0][1:])
                subject = int(naming_list[1][1:])
                edition = int(naming_list[2][1:])

                self.clips.append(os.path.join(root, file))
                self.labels.append(action)
                self.subjects.append(subject)
                self.edition.append(edition)

                if self.get_metadata:
                    #read in the clip and get the length
                    len_current_clip = self.__getitem__(len(self.clips)-1)[0].shape[1]
                    self.meta_data["clip_lengths"].append(len_current_clip)


        print("#########################################")
        print(len(self.clips))
        print("#########################################")
        # print min max and quartiels of clip lengths
        if self.get_metadata:
            print("min clip length: ", np.min(self.meta_data["clip_lengths"]))
            print("25th percentile: ", np.percentile(self.meta_data["clip_lengths"], 25))
            print("50th percentile: ", np.percentile(self.meta_data["clip_lengths"], 50))
            print("75th percentile: ", np.percentile(self.meta_data["clip_lengths"], 75))
            print("max clip length: ", np.max(self.meta_data["clip_lengths"]))

            #save clip lengths to file
            np.save("/data2/fhuemer/MSRActivity3D/metadata/clip_lengths.npy", self.meta_data["clip_lengths"])

       
   
    def subject_selector(self):
        subjects = list(range(1,10))
        if self.mode == "train": 
            #leave out every third subject
            self.chosen_subjects = [x for x in subjects if x % 3 != 0]
        elif self.mode == "test":
            raise NotImplementedError
            self.chosen_subjects = subjects[2::3]
        elif self.mode == "val":
            self.chosen_subjects = subjects[2::3]

    def repeat_or_cutoff(self, clip):
        #repeat or cut off frames to max_number_frames
        if clip.shape[0] < self.max_number_frames:
            clip = np.repeat(clip, math.ceil(self.max_number_frames/clip.shape[0]) + 1 , axis=0)
            clip = clip[:self.max_number_frames, :, :, :]
        elif clip.shape[0] > self.max_number_frames:
            clip = clip[:self.max_number_frames, :, :, :]
        return clip

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        # returns sample and label for index 
        clip_path = self.clips[idx]
        label = self.labels[idx]
        clip = load_depth_map(clip_path)

        #add chanel dimension
        clip = np.expand_dims(clip, axis=3)

        #repeat or cut off frames to max_number_frames to get even number of frames
        clip = self.repeat_or_cutoff(clip)
        
        #here the dimensons get rearranged to fit the pytorch format. Watch out! 
        if self.transform:
            clip = self.transform(clip)

        return clip, label 

def read_header(file):
    # Read the number of frames
    num_frames = np.frombuffer(file.read(4), dtype=np.uint32)[0]
    # Read the dimensions of the frames
    dims = np.frombuffer(file.read(8), dtype=np.uint32)
    return dims, num_frames

def load_depth_map(path):
    with open(path, 'rb') as file:
        # Read header
        dims, num_frames = read_header(file)
        # Read the rest of the file data as uint32
        file_data = np.fromfile(file, dtype=np.uint32)
    
    # Convert to depth map format
    depth_count_per_map = np.prod(dims)
    depth_maps = []

    for i in range(num_frames):
        current_depth_data = file_data[:depth_count_per_map]
        file_data = file_data[depth_count_per_map:]
        depth_map = current_depth_data.reshape((dims[1], dims[0])).T
        depth_maps.append(depth_map)

    depth_maps = np.stack(depth_maps, axis=0)
    return depth_maps





       
        

