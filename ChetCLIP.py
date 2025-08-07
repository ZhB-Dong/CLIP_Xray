from ChetXpertDataloader import ChestXrayDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import clip
import os
import pandas as pd

csv_path = 'E:/codeDZB/MyDeepLearning/CLIP/Datasets/ChestXray/Data_Entry_2017_v2020.csv'
image_dir = 'E:/codeDZB/MyDeepLearning/CLIP/Datasets/ChestXray/images_001/images/'

def custom_collate_fn(batch):
    images, labels = zip(*batch)  # unzip list of tuples
    images = torch.stack(images, dim=0)  # 正常堆叠图像
    return images, labels  # 保持 labels 是 list of lists（不等长）


def gene_prompt(labels):
    if labels == 'No Finding':
        return 'No Finding'
    else:
        prompts = ', '.join(labels[0])
        return prompts

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ChestXrayDataset(
    csv_path=csv_path,
    image_dir=image_dir,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

device = 'cpu'
model, preprocess = clip.load('ViT-B/32', device)

data = pd.read_csv(csv_path)

# 获取你真实拥有的图像名集合（加快过滤）
existing_images = set(os.listdir(image_dir))

# 过滤：只保留图像存在的数据行
csvdata = data[data['Image Index'].isin(existing_images)].reset_index(drop=True)

# 构建标签映射（可选：多标签 one-hot）
label_set = sorted(set(
    label for row in csvdata['Finding Labels'].dropna().str.split('|') for label in row
))
labels_all = ['Brain', 'Chest', 'Heart']
text_inputs = torch.cat([clip.tokenize(f"a X-ray of a {c}") for c in labels_all]).to(device)

right_count = 0
cnt = 1
# 测试一批数据
for images, labels in dataloader:
    # print(cnt)

    if cnt > 1000:
        break
    if cnt % 100 == 0:
        print(cnt)
    cnt = cnt+1
    # print(images.shape)  # torch.Size([8, 3, 224, 224])
    # print(labels)        # [['No Finding'], ['Cardiomegaly', 'Effusion'], ...]
    # print('Ground Truth: ' + gene_prompt(labels))
    # print(gene_prompt(labels))
    image_input = images.to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    if indices.item() == 1:
        right_count += 1
    # print(labels_all[indices.item()])
    # print(indices)
    # print("Top predictions:")
    # for value, index in zip(values, indices):
    #     print(f"{labels_all[index]:>16s}: {100 * value.item():.2f}%")
    #     if value.item() in labels:
    #         right_count += 1
    # break

print(right_count/cnt)
