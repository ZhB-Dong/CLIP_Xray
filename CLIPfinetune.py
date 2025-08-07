from BodyPartDataset import ClassificationDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import clip
import os
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from datetime import datetime
# from loguru import logger

root_path = '/home/sda2/dzb/Datasets/BodyPartsX-RayImages/'
csv_path = '/home/sda2/dzb/Datasets/BodyPartsX-RayImages/./train_df.csv'
Labels = ['Abdomen', 'Ankle', 'Cervical Spine', 'Chest', 'Clavicles', 'Elbow', 'Feet',
          'Finger', 'Forearm', 'Hand', 'Hip', 'Knee', 'Lower Leg', 'Lumbar Spine', 'Others', 'Pelvis', 'Shoulder',
          'Sinus', 'Skull', 'Thigh', 'Thoracic Spine', 'Wrist']
epoches = 50
lr = 1e-6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, preprocess = clip.load("ViT-B/32", device=device, jit=False)

optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1)

# 创建损失函数
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# 加载数据集
your_dataset = ClassificationDataset(
    csv_path=csv_path,
    root_dir=root_path,
    labels=Labels,
    transform=preprocess,
    idx_range=[1,1200]
)

test_dataset = ClassificationDataset(
    csv_path=csv_path,
    root_dir=root_path,
    labels=Labels,
    transform=preprocess,
    idx_range=[1200,1600]
)

dataset_size = len(your_dataset)
your_dataloader = DataLoader(your_dataset, batch_size=16, shuffle=True, pin_memory=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False)

phase = "train"
model_name = "CLIP_FT1"
ckt_gap = 4
now = datetime.now()
dirname = f'/home/sda2/dzb/Weights/CLIP_0807/{now.strftime("%Y-%m-%d_%H%M%S")}_clip_weight'
# if dirname in os.listdir() is False:
os.makedirs(dirname)
text_inputs = torch.cat([clip.tokenize(f"a X-Ray image of {c} part") for c in Labels]).to(device) # 0.6253
test_cnt = 1
right_cnt = 0
for epoch in range(0, epoches):
    scheduler.step()
    total_loss = 0
    batch_num = 0
    # 使用混合精度，占用显存更小
    with torch.cuda.amp.autocast(enabled=True):
        for images, label in your_dataloader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = clip.tokenize(label).to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                logits_per_image, logits_per_text = net(images, label_tokens)
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_loss += cur_loss
                if phase == "train":
                    cur_loss.backward()
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        clip.model.convert_weights(net)
            # if batch_num % 4 == 0:
            # print('{} epoch:{} loss:{:.4f}'.format(phase,epoch,cur_loss))
            # logger.info('{} epoch:{} loss:{}'.format(phase, epoch, cur_loss))
        epoch_loss = total_loss/dataset_size
        torch.save(net.state_dict(), dirname+f"/{model_name}_epoch_{epoch}.pth")
        print(f"weights_{epoch} saved")
        if epoch % ckt_gap == 0:
            checkpoint_path = dirname+f"/{model_name}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            print(f"checkpoint_{epoch} saved")
            print('{} Loss: {}'.format(phase, epoch_loss))

    for images, labels in test_dataloader:
        # print(labels)
        # print(cnt)

        # if cnt > 300:
        #     break
        test_cnt = test_cnt + 1
        image_input = images.to(device)
        with torch.no_grad():
            image_features = net.encode_image(image_input)
            text_features = net.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        for indice in indices:
            if labels[0] == Labels[indice.item()]:
                right_cnt += 1
        # print(text_features[indices.item()])
        # print(indices)
        # print("Top predictions:")
        # for value, index in zip(values, indices):
        #     print(f"{Labels[index]:>16s}: {100 * value.item():.2f}%")
        # if value.item() in labels:
        # right_count += 1
        # break
    # if epoch % ckt_gap == 0:
    print('weights_{} Test right rate {:.4f}'.format(epoch, right_cnt / test_cnt))
