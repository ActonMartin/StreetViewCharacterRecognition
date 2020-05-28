import glob
import os

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import transforms

from dataset_load.dataset import SVHNDataset
from model.model import SVHN_Model1
from model.model import predict
from utils.seed import init_seeds

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

init_seeds(0)
model = SVHN_Model1().cuda()
test_path = sorted(glob.glob('D:/Projects/wordec/input/test/*.png'))
# test_json = json.load(open('../input/test.json'))
test_label = [[1]] * len(test_path)
# print(len(test_path), len(test_label))

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((70, 140)),
                    # transforms.RandomCrop((60, 120)),
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

# 加载保存的最优模型
model.load_state_dict(torch.load('D:/Projects/wordec/model.pt'))

test_predict_label = predict(test_loader, model, 1)
print(test_predict_label.shape)
print('test_predict_label', test_predict_label)

test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
# print('test_label', test_label)
test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1),
    test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1),
    test_predict_label[:, 33:44].argmax(1),
    test_predict_label[:, 44:55].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x != 10])))
# print("test_label_pred", len(test_label_pred))
df_submit = pd.read_csv('D:/Projects/wordec/input/test_A_sample_submit.csv')
df_submit['file_code'] = test_label_pred
df_submit.to_csv('submit_1018.csv', index=None)
print("finished")
