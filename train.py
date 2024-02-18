import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io, transform
from skimage.color import rgb2gray
from PIL import Image
import sys
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import copy
from tqdm import tqdm
import cv2
from network import Net
from torch.nn import DataParallel
from dataset import BrainDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau

BASE_DIR = 'results'
OUT_DIR = 'brainMRI'

if not os.path.exists(os.path.join(BASE_DIR, OUT_DIR)):
    os.mkdir(os.path.join(BASE_DIR, OUT_DIR))

writer = SummaryWriter(log_dir=f'./runs/{OUT_DIR}')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cpu")
transformed_dataset = BrainDataset(datasets=["./images/train"])
# Training dataset
train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=2, shuffle=True, num_workers=16)

val_loader = torch.utils.data.DataLoader(BrainDataset(datasets=["./images/val"], batch_size=2, shuffle=True, num_workers=1))

model = Net()
# model.load_state_dict(torch.load('/scratch0/Palmprint/keypoint_detection/results/zk_kpt_child_adult_tongji_resnet_perspective_transform_more_rotation_more_scaling_finetune/94.pt'))
model.to(device)
# model = DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min')
# scheduler2 = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
loss_fn = nn.Softmax()

train_results = os.path.join('results', OUT_DIR, 'train')
test_results = os.path.join('results', OUT_DIR, 'val')

if not os.path.exists(train_results):
    os.makedirs(train_results)

if not os.path.exists(test_results):
    os.makedirs(test_results)

save_image_dir = os.path.join(test_results, 'images')
if not os.path.isdir(save_image_dir):
    os.makedirs(save_image_dir)

previous_loss = 10**50
average_test_loss = 10**50
for epoch in range(1, 201):
    model.train()
    for batch_idx, sampled_batch in enumerate(train_loader):
        data, target, image_name, image = sampled_batch
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        target = torch.squeeze(target)
        loss = loss_fn(output, target)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        writer.add_scalar('Loss/train', loss, epoch*batch_idx)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    torchvision.utils.save_image(
            data,
            os.path.join(test_results, 'images', 'train-batch-%d.jpg' % epoch),
            padding=0,
            normalize=True) 

    scheduler.step(loss)
    torch.cuda.empty_cache()
    print()
    print('computing validation loss...\n')

    with torch.no_grad():
        model.eval()
        total_loss = 0
        count = 0
        max_img = 5
        for b_i, sample in enumerate(val_loader):
            data, target, image_name, image = sample
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            target = torch.squeeze(target)
            val_loss = loss_fn(output, target)

            if val_loss.item() < previous_loss:

                print('val loss decreased... saving model')
                previous_loss = val_loss.item()

            # if count <= max_img:
                im = image[0].detach()
                kpts = target[0].detach().cpu().numpy().astype(int)
                img = np.array(Ft.to_pil_image(im))[:, :, ::-1].copy()
                for i, k in enumerate(kpts):
                    img = cv2.circle(img, (k[0], k[1]), radius=1, color=(0,0,255), thickness=-1)
                cv2.imwrite(os.path.join(test_results, f'groundtruth_{epoch}_{count}.jpg'), img)

                kpts = output[0].detach().cpu().numpy().astype(int)
                img = np.array(Ft.to_pil_image(im))[:, :, ::-1].copy()
                for i, k in enumerate(kpts):
                    img = cv2.circle(img, (k[0], k[1]), radius=1, color=(0,0,255), thickness=-1)
                cv2.imwrite(os.path.join(test_results, f'predicted_{epoch}_{count}.jpg'), img)
                torch.save(model.state_dict(), f'{BASE_DIR}/{OUT_DIR}/{epoch}.pt')
            total_loss += val_loss.item()
            count += 1
            break

        print('average test loss:', round(total_loss/count, 2))


writer.flush()
writer.close()