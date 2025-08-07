from BodyPartDataset import ClassificationDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import clip
import os
import pandas as pd
import numpy as np

# root_path = 'E:/codeDZB/MyDeepLearning/CLIP/Datasets/BodyPartsX-RayImages/'
# csv_path = 'E:/codeDZB/MyDeepLearning/CLIP/Datasets/BodyPartsX-RayImages/./train_df.csv'

root_path = '/home/sda2/dzb/Datasets/BodyPartsX-RayImages/'
csv_path = '/home/sda2/dzb/Datasets/BodyPartsX-RayImages/./train_df.csv'
# csv_path = '/home/sda2/dzb/Datasets/BodyPartsX-RayImages/./test_df.csv'
weight_patch = '/home/sda2/dzb/Weights/CLIP_0807/2025-08-07_115948_clip_weight/CLIP_FT1_epoch_49.pth'

Labels = ['Abdomen', 'Ankle', 'Cervical Spine', 'Chest', 'Clavicles', 'Elbow', 'Feet',
          'Finger', 'Forearm', 'Hand', 'Hip', 'Knee', 'Lower Leg', 'Lumbar Spine', 'Others', 'Pelvis', 'Shoulder',
          'Sinus', 'Skull', 'Thigh', 'Thoracic Spine', 'Wrist']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # torch.numpy()
])
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device) # 0.62
model.load_state_dict((torch.load(weight_patch, map_location=device))) # 0.78

dataset = ClassificationDataset(
    csv_path=csv_path,
    root_dir=root_path,
    labels=Labels,
    transform=preprocess,
    idx_range=[1200,1600]
)


dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

text_inputs = torch.cat([clip.tokenize(f"a X-Ray image of {c} part") for c in Labels]).to(device) # 0.6253
# text_inputs = torch.cat([clip.tokenize(f"a X-Ray image of {c} part in human body") for c in Labels]).to(device) # 0.6049
# text_inputs = torch.cat([clip.tokenize(f"a X-ray of a {c}") for c in Labels]).to(device) # 0.6030
# text_inputs = torch.cat([clip.tokenize(f"This is a X-ray image of {c} part. ") for c in Labels]).to(device) # 0.5887
# text_inputs = torch.cat([clip.tokenize(f"a X-ray image of {c} of human") for c in Labels]).to(device) # 0.5974
# text_inputs = torch.cat([clip.tokenize(f"a image of {c}") for c in Labels]).to(device) # 0.5650

right_count = 0
cnt = 1
# 测试一批数据
for images, labels in dataloader:
    # print(cnt)

    # if cnt > 300:
    #     break
    if cnt % 10 == 0:
        print(cnt)
    cnt = cnt+1
    # print(images.shape)  # torch.Size([8, 3, 224, 224])
    # print(Labels[labels.item()])        # [['No Finding'], ['Cardiomegaly', 'Effusion'], ...]
    # print('Ground Truth: ' + Labels[labels.item()])
    # print(gene_prompt(labels))
    image_input = images.to(device)
    # print(images.type())
    # image_input = preprocess(images).unsqueeze(0).to(device)
    # print(image_input.size())
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    for indice in indices:
        if labels[0] == Labels[indice.item()]:
            right_count += 1
    # print(text_features[indices.item()])
    # print(indices)
    # print("Top predictions:")
    # for value, index in zip(values, indices):
    #     print(f"{Labels[index]:>16s}: {100 * value.item():.2f}%")
        # if value.item() in labels:
            # right_count += 1
    # break

print(right_count/cnt)
