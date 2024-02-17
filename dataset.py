import os
from glob import glob
import torchvision
import torch
from torch.utils.data import Dataset
from torch.utils import data
from torchvision import transforms as T
from collections import defaultdict
from PIL import Image

class BrainDataset(Dataset):
    def __init__(self, datasets):
        self.imgs = []
        self.labels = []
        self.imgs_dict = defaultdict(list)

        for dataset in datasets:
            images = glob(f"{dataset}/*/*.jpg")
            self.imgs += images

        normalize_gray = T.Normalize(mean=[0.456], std=[0.224])
        
        for image in self.imgs:
            img_name = image.split('/')[-1]
            label = 1 if 'yes' in img_name else 0
            self.labels.append(label)
            self.imgs_dict[label].append(image)
        
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        
        self.transform = T.Compose([T.ToTensor(), normalize])
    
    def __get_item__(self, idx):

        img_path = self.imgs[idx]
        if self.input_shape[0] == 1:
            data = Image.open(img_path).convert('L')
        else:
            data = Image.open(img_path).convert('RGB')
        
        data = self.transform(data)
        label = self.labels[idx]

        return data.float(), label

if __name__ == "__main__":
    dataset = BrainDataset(datasets='augmented')

    trainloader = data.DataLoader(dataset, batch_size=20, shuffle=False)
    



