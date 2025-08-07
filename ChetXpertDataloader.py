import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # 加载 CSV 数据
        data = pd.read_csv(csv_path)

        # 获取你真实拥有的图像名集合（加快过滤）
        existing_images = set(os.listdir(image_dir))

        # 过滤：只保留图像存在的数据行
        self.data = data[data['Image Index'].isin(existing_images)].reset_index(drop=True)

        # 构建标签映射（可选：多标签 one-hot）
        self.label_set = sorted(set(
            label for row in self.data['Finding Labels'].dropna().str.split('|') for label in row
        ))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['Image Index']
        label_str = row['Finding Labels']

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = label_str.split('|') if pd.notna(label_str) else []
        multi_hot = torch.zeros(len(self.label_set))
        for label in labels:
            if label in self.label_to_idx:
                multi_hot[self.label_to_idx[label]] = 1.0

        return image, labels