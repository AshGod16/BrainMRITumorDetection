import os
from glob import glob
from tqdm import tqdm
import cv2

def resize_imgs(img_paths, out='images'):

    if not os.path.exists(out):
        os.makedirs(out)
    for image in tqdm(img_paths):
        img_name = image.split('/')[-1]
        img = cv2.imread(image)
        img = cv2.resize(img, (224,224))
        cv2.imwrite(os.path.join(out, img_name), img)

images = glob('./augmented_data/*/*.jpg')

print("images", len(images))

yes = [i for i in images if 'yes' in i]
no = [i for i in images if 'no' in i]

train_yes, val_yes, test_yes = yes[:int(0.8*len(yes))], yes[int(0.8*len(yes)):int(0.9*len(yes))], yes[int(0.9*len(yes)):]
train_no, val_no, test_no = no[:int(0.8*len(no))], no[int(0.8*len(no)):int(0.9*len(no))], no[int(0.9*len(no)):]

resize_imgs(train_yes, "images/train")
resize_imgs(train_no, "images/train")

resize_imgs(val_yes, "images/val")
resize_imgs(val_no, "images/val")

resize_imgs(test_yes, "images/test")
resize_imgs(test_no, "images/test")
