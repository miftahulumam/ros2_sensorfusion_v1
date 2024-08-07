import torch
import torch.nn as nn

from .mono_up import MonoUp

class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()

        self.stereo_input = ['a', 'b', 'c', 'd', 'e', 'f']
        self.mono_input = ['a', 'b', 'c', 'd', 'e', 'f', 'MonoUp']
        self.model_name = model_name

        if model_name == 'MonoUp':
            self.model = MonoUp()

    
    def forward(self, x2, x3):
        disp_4, disp_3, disp_2, disp_1 = 0, 0, 0, 0

        if self.model_name in self.stereo_input:
            disp_4, disp_3, disp_2, disp_1 = self.model(x2, x3)
        elif self.model_name in self.mono_input:
            disp_4, disp_3, disp_2, disp_1 = self.model(x2)

        return disp_4, disp_3, disp_2, disp_1