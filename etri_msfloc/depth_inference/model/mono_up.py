import torch
import torch.nn as nn

from .all_attention import ACmix, CBAMBlock

class Conv3x3Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride, padding):
        super(Conv3x3Block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, padding)
        self.BN = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p=0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.BN(x)
        x = self.dropout(x)
        return x

class MultiConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=0):
        super(MultiConvBlock, self).__init__()
        self.convBlock_1 = Conv3x3Block(in_ch, out_ch, stride, padding)
        self.convBlock_2 = Conv3x3Block(out_ch, out_ch, stride, padding)

    def forward(self, x):
        x = self.convBlock_1(x)
        x = self.convBlock_2(x)
        return x

class UpsampleJoinBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpsampleJoinBlock, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(in_ch, in_ch, 3, 2)
        self.conv_2 = nn.Conv2d(in_ch+out_ch, out_ch, 3, 1, 'same')

    def forward(self, x1, x2):
        # print(f'upsample before | x1 {x1.shape} | x2 {x2.shape} | {x1.shape[1]} | {x1.shape[2]}' )
        x1 = self.conv_1(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = nn.ZeroPad2d(padding=(diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))(x1)

        # print(f'upsample | x1 {x1.shape} | x2 {x2.shape} |' )
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_2(x)

        return x
    
class DisparityOut(nn.Module):
    def __init__(self, in_ch):
        super(DisparityOut, self).__init__()
        self.conv_1 = MultiConvBlock(in_ch, 2, 1, 'same')
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)

        return x
    
class MonoUp(nn.Module):
    def __init__(self, batchnorm=True):
        super(MonoUp, self).__init__()
        self.conv_1 = MultiConvBlock(3, 32, 1, 'same') #in 192x640 | out 192x640
        self.half_1 = nn.Conv2d(32, 64, 3, 2, 1) #in 192x640 | out 86x320
        self.pool_1 = nn.AvgPool2d(3, 2, 1) #in 192x640 | out 86x320

        self.conv_2 = MultiConvBlock(64+3, 64, 1, 'same') #in 86x320 | out 86x320
        self.att_2 = CBAMBlock(64) #in 86x320 | out 86x320
        self.half_2 = nn.Conv2d(64, 128, 3, 2, 1) #in 86x320 | out 43x160
        self.pool_2 = nn.AvgPool2d(3, 4) #in 192x640 | out 43x160

        self.conv_3 = MultiConvBlock(128+3, 128, 1, 'same') #in 43x160 | out 43x160
        self.att_3 = CBAMBlock(128) #in 43x160 | out 43x160
        self.half_3 = nn.Conv2d(128, 256, 3, 2, 1) #in 43x160 | out 22x80
        self.pool_3 = nn.AvgPool2d(3, 8) #in 192x640 | out 22x80

        self.conv_4 = MultiConvBlock(256+3, 256, 1, 'same') #in 43x160 | out 43x160
        self.att_4 = CBAMBlock(256) #in 43x160 | out 43x160
        self.half_4 = nn.Conv2d(256, 512, 3, 2, 1) #in 43x160 | out 22x80
        self.pool_4 = nn.AvgPool2d(3, 16) #in 192x640 | out 22x80

        self.bottom_1 = MultiConvBlock(512+3, 512, 1, 'same')
        self.bottom_2 = MultiConvBlock(512, 512, 1, 'same')

        self.up_4 = UpsampleJoinBlock(512, 256)
        self.conv_5 = MultiConvBlock(256, 256, 1, 'same')
        self.disparity_5 = DisparityOut(256)

        self.up_3 = UpsampleJoinBlock(256, 128)
        self.conv_6 = MultiConvBlock(128, 128, 1, 'same')
        self.disparity_6 = DisparityOut(128)

        self.up_2 = UpsampleJoinBlock(128, 64)
        self.conv_7 = MultiConvBlock(64, 64, 1, 'same')
        self.disparity_7 = DisparityOut(64)

        self.up_1 = UpsampleJoinBlock(64, 32)
        self.conv_8 = MultiConvBlock(32, 3, 1, 'same')
        self.disparity_8 = DisparityOut(3)

    def forward(self, x_left):
        x_1 = self.conv_1(x_left)
        x_half_1 = self.half_1(x_1)
        x_avg_1 = self.pool_1(x_left)
        x_avg_1 = self.gauss_noise_tensor(x_avg_1, 1)
        # print('x_1 sahpe, x_half_1 sahpe, x_avg_1 sahpe', x_1.shape, x_half_1.shape, x_avg_1.shape)

        x_2 = torch.cat((x_half_1, x_avg_1), dim=1)
        # print('x2 sahpe', x_2.shape)
        x_2 = self.conv_2(x_2)
        # print('x2 sahpe', x_2.shape)
        x_2 = self.att_2(x_2)
        x_half_2 = self.half_2(x_2)
        x_avg_2 = self.pool_2(x_left)
        x_avg_2 = self.gauss_noise_tensor(x_avg_2, 2)

        x_3 = torch.cat((x_half_2, x_avg_2), dim=1)
        x_3 = self.conv_3(x_3)
        x_3 = self.att_3(x_3)
        x_half_3 = self.half_3(x_3)
        x_avg_3 = self.pool_3(x_left)
        x_avg_3 = self.gauss_noise_tensor(x_avg_3, 3)

        x_4 = torch.cat((x_half_3, x_avg_3), dim=1)
        x_4 = self.conv_4(x_4)
        x_4 = self.att_4(x_4)
        x_half_4 = self.half_4(x_4)
        x_avg_4 = self.pool_4(x_left)
        x_avg_4 = self.gauss_noise_tensor(x_avg_4, 4)

        x_bottom = torch.cat((x_half_4, x_avg_4), dim=1)
        x_bottom = self.bottom_1(x_bottom)
        x_bottom = self.bottom_2(x_bottom)

        x_5 = self.up_4(x_bottom, x_4)
        x_5 = self.conv_5(x_5)
        x_disparity_5 = self.disparity_5(x_5)

        x_6 = self.up_3(x_5, x_3)
        x_6 = self.conv_6(x_6)
        x_disparity_6 = self.disparity_6(x_6)

        x_7 = self.up_2(x_6, x_2)
        x_7 = self.conv_7(x_7)
        x_disparity_7 = self.disparity_7(x_7)

        x_8 = self.up_1(x_7, x_1)
        x_8 = self.conv_8(x_8)
        x_disparity_8 = self.disparity_8(x_8)

        # print('Disparity out size: ', x_disparity_5.shape, x_disparity_6.shape, x_disparity_7.shape, x_disparity_8.shape)

        return x_disparity_8, x_disparity_7, x_disparity_6, x_disparity_5

    # adapted from https://github.com/pytorch/vision/issues/6192
    def gauss_noise_tensor(self, img, level):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        
        if level == 1:
            sigma = 25.0
        elif level == 2:
            sigma = 50.0
        elif level == 3:
            sigma = 75.0
        elif level == 4:
            sigma = 99.0
        
        out = img + sigma * torch.randn_like(img)
        
        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out