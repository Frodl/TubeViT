import torch

class repeat_color_channel():
    def __call__(self, clip):
        return clip.repeat(3, 1, 1, 1)
    

class min_max_normalization():
    def __init__(self, scale_up=False):
        self.scale_up = scale_up

    def __call__(self, clip):   
        clip = clip.astype(float)
        clip = clip - clip.min()
        if clip.max() != 0:
            clip = clip / clip.max()
        if self.scale_up:
            clip = clip * 255       
        return clip   


class ConvertToUint8(object):
    def __call__(self, clip):
        return torch.from_numpy(clip).to(torch.uint8)

class ConvertToFloat32(object):
    def __call__(self, clip):
        return torch.from_numpy(clip).to(torch.float32) 
    

class ConvertToFloat64(object):
    def __call__(self, clip):
        return torch.from_numpy(clip).to(torch.float64) 
    
class sample_frames():
    def __init__(self, nth):
        self.nth = nth

    def __call__(self, clip):
        return clip[:, ::self.nth, :, :]
    

class PermuteDimensions:
    def __init__(self, order):
        self.order = order

    def __call__(self, x):
        return x.permute(self.order)