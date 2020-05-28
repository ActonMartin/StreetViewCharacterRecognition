import os
import time

import numpy as np
import torch
import torch.nn as nn

from dataset_load.dataloader import train_loader
from dataset_load.dataloader import val_loader
from model.model import SVHN_Model1
from model.model import predict
from model.model import train
from model.model import validate
from utils.seed import init_seeds

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

init_seeds(0)
model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 1000.0

use_cuda = True
if use_cuda:
    model = model.cuda()

for epoch in range(50):
    start = time.time()
    print('start', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)))
    train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate(val_loader, model, criterion)

    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x != 10])))

    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
    end = time.time()
    print('end', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end)))
    time_cost = end - start
    print(
        'Epoch: {0}, Train loss: {1} \t Val loss: {2}, time_cost: {3}'.format(
            epoch,
            train_loss,
            val_loss,
            time_cost))
    print('Val Acc', val_char_acc)
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        # print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model.state_dict(), './model.pt')
