import torch

class repeat_color_channel():
    def __call__(self, video):
        return video.repeat(3, 1, 1, 1)
    

class min_max_normalization():
    def __init__(self, scale_up=True):
        self.scale_up = scale_up

    def __call__(self, video):   
        #video = video.float()
        video = video - video.min()
        if video.max() != 0:
            video = video / video.max()
        if self.scale_up:
            video = video * 255       
        return video   


class ConvertToUint8(object):
    def __call__(self, clip):
        return torch.from_numpy(clip).to(torch.uint8)

class ConvertToFloat32(object):
    def __call__(self, clip):
        return torch.from_numpy(clip).to(torch.float32) 
    

class sample_frames():
    def __init__(self, nth):
        self.nth = nth

    def __call__(self, video):
        return video[:, ::self.nth, :, :]