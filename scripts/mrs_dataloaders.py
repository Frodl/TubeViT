### Dataloaders for ETRI dataset ###
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import cv2
# train = 2/3
# val = 1/6
# test = 1/6

class MRSActivityDataset(Dataset):

    def __init__(self, root_dir,max_number_frames, remove_background = False,
                  mode = "train" ,transform=None, eval_mode = "cs", get_metadata = False) -> None:
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
        self.remove_background = remove_background
        self.subject_selector()

        #walk through root_dir and create list of images and labels
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                naming_list = file.split('_')

                action = int(naming_list[0][1:])-1
                subject = int(naming_list[1][1:])-1
                edition = int(naming_list[2][1:])-1

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

        print(f"Mode: {self.mode}, Chosen Subjects: {self.chosen_subjects}")
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
        clip = self.load_depth_map(clip_path)

        #add chanel dimension
        clip = np.expand_dims(clip, axis=3)

        #repeat or cut off frames to max_number_frames to get even number of frames
        clip = self.repeat_or_cutoff(clip)
        
        #here the dimensons get rearranged to fit the pytorch format. Watch out! 
        if self.transform:
            clip = self.transform(clip)

        return clip, label 

    def load_depth_map(self, depth_file, resize='VGA'):
        ''' Extracts depth images and masks from the MSR Daily Activites dataset
        ---Parameters---
        depth_file : filename for set of depth images (.bin file)
        '''

        file_ = open(depth_file, 'rb')

        ''' Get header info '''
        frames = np.frombuffer(file_.read(4), dtype=np.int32)[0]
        cols = np.frombuffer(file_.read(4), dtype=np.int32)[0]
        rows = np.frombuffer(file_.read(4), dtype=np.int32)[0]

        ''' Get depth/mask image data '''
        data = file_.read()

        '''
        Depth images and mask images are stored together per row.
        Thus we need to extract each row of size n_cols+n_rows
        '''
        dt = np.dtype([('depth', np.int32, cols), ('mask', np.uint8, cols)])

        ''' raw -> usable images '''
        frame_data = np.frombuffer(data, dtype=dt)
        depthIms = frame_data['depth'].astype(np.uint16).reshape([frames, rows, cols])
        maskIms = frame_data['mask'].astype(np.uint16).reshape([frames, rows, cols])


        depthIms = np.stack([cv2.resize(depthIms[d,:,:], (640,480)) for d in range(len(depthIms))],0)
        maskIms = np.stack([cv2.resize(maskIms[d,:,:], (640,480)) for d in range(len(maskIms))],0)

        if self.remove_background:
            binary_mask = (maskIms > 0).astype(np.uint16)
            depthIms = depthIms * binary_mask	
        depthIms = np.flip(depthIms, 1)
        
        return depthIms





        
            

