### Dataloaders for ETRI dataset ###
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import zipfile


# train = 2/3
# val = 1/6
# test = 1/6

class ETRIDataset(Dataset):

    def __init__(self, root_dir,max_number_frames, mode = "train", remove_background = True ,transform=None, eval_mode = "cs", single_camera= False, elders_only = False, get_metadata = False):
        """
        Args:
            root_dir (string): Directory with all the clips.
            mode (string): train, test, or val
            remove_background (bool): remove background from images, or use cropped images 
            transform (callable, optional): Optional transform to be applied on a sample.
            eval_mode (string): for now only cs (cross subject), maybe later cv (cross view)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.remove_background = remove_background
        self.subject_selector()
        self.labels =[]
        self.clips = []
        self.room = []
        self.camera = []
        self.max_number_frames = max_number_frames
        self.get_metadata = get_metadata
        self.meta_data = {
            "clip_lengths": [],
            "filenames": [],
            "labels": [],}
        self.corrupted_files = []


        #walk through root_dir and create list of images and labels
        for root, dirs, files in os.walk(root_dir):
            if len(dirs) > 0:
                for dir in dirs:
                    if dir.startswith("A"):
                        naming_list = dir.split('_')
                        action = naming_list[0]
                        subject = naming_list[1]
                        room = naming_list[2]
                        camera = naming_list[3]                      
                        if (single_camera and camera == "C002") or not single_camera:
                                if (elders_only and int(subject[1:]) <= 50) or not elders_only:
                                    if int(subject[1:]) in self.chosen_subjects:
                                        self.clips.append(os.path.join(root, dir))
                                        self.labels.append(int(action[1:]) - 1)
                                        self.room.append(room)
                                        self.camera.append(camera)

                                        if self.get_metadata:
                                            #read in the clip and get the length
                                            len_current_clip = self.__getitem__(len(self.clips)-1)[0].shape[1]
                                            self.clip_lengths.append(len_current_clip)


        print("#########################################")
        print(len(self.clips))
        print("#########################################")
        # print min max and quartiels of clip lengths
        if self.get_metadata:
            print("min clip length: ", np.min(self.clip_lengths))
            print("25th percentile: ", np.percentile(self.clip_lengths, 25))
            print("50th percentile: ", np.percentile(self.clip_lengths, 50))
            print("75th percentile: ", np.percentile(self.clip_lengths, 75))
            print("max clip length: ", np.max(self.clip_lengths))

            #save clip lengths to file
            np.save("/data2/fhuemer/etri/meta_data/clip_lengths.npy", self.clip_lengths)

       
   
    def subject_selector(self):
        subjects = list(range(1,100))
        if self.mode == "train": 
            #leave out every third subject
            self.chosen_subjects = [x for x in subjects if x % 3 != 0]
        elif self.mode == "test":
            self.chosen_subjects = subjects[2::6]
        elif self.mode == "val":
            self.chosen_subjects = subjects[5::6]

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
        clip_path = os.path.join(clip_path, os.path.basename(clip_path) + ".npz")
        label = self.labels[idx]
        
        clip = np.load(clip_path)
            
        #extracting the array from the npz file
        clip = clip.f.arr_0

        halfwaypoint = clip.shape[0]//2
        if self.remove_background:
            clip = clip[:halfwaypoint, :, :, :]
        else:
            clip = clip[halfwaypoint:, :, :, :]
        
        #repeat or cut off frames to max_number_frames to get even number of frames
        clip = self.repeat_or_cutoff(clip)
        

        #here the dimensons get rearranged to fit the pytorch format. Watch out! 
        if self.transform:
            clip = self.transform(clip)

        return clip, label
    