import glob
import json
import os

import torch
import torchvision.transforms as transforms

from dataset_load.dataset import SVHNDataset
from utils.seed import init_seeds

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

init_seeds(0)
train_path = sorted(glob.glob('D:/Projects/wordec/input/train/*.png'))
train_json = json.load(open('D:/Projects/wordec/input/train.json'))
train_label = [train_json[x]['label'] for x in train_json]
print(len(train_path), len(train_label))

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=40,
    shuffle=True,
    num_workers=0,
)

val_path = sorted(glob.glob('E:/CLONE/wordec/input/val/*.png'))
val_json = json.load(open('E:/CLONE/wordec/input/val.json'))
val_label = [val_json[x]['label'] for x in val_json]
print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((60, 120)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [
                                         0.229, 0.224, 0.225])
                ])),
    batch_size=40,
    shuffle=False,
    num_workers=0,
)
