import torch
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

from model.net import Net

device = 'cuda:1'

cap = cv2.VideoCapture(0)

model_wt_path = '/home/krishna/HDD6_1/DepthEstimation/best_model/best_val_loss.pth'

rsz_h = 192
rsz_w = 640

model = Net('MonoUp')
model = model.to(device)
model.load_state_dict(torch.load(model_wt_path))
model.eval()

while True:
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (rsz_w, rsz_h))
    frame = frame / 255.
    frame = torch.tensor(frame, dtype=torch.float32, device=device)
    frame = frame.permute(2,0,1)
    frame = torch.unsqueeze(frame, 0)

    output = model(frame, frame)
    
    plt.imshow(output[0][0][0].detach().cpu().numpy(), cmap='magma')
    plt.pause(.8)
    
cap.release()